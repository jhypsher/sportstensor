import os
import time
from dotenv import load_dotenv
import bittensor as bt
from base.miner import BaseMinerNeuron
from common.data import League, get_league_from_string
from common.protocol import GetLeagueCommitments, GetMatchPrediction
from st.sport_prediction_model import make_match_prediction
from huggingface_hub import hf_hub_download
from keras.models import load_model

# Define the path to the miner.env file
MINER_ENV_PATH = os.path.join(os.path.dirname(__file__), 'miner.env')
load_dotenv(dotenv_path=MINER_ENV_PATH, override=True)

class Miner(BaseMinerNeuron):
    """The Sportstensor Miner."""

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)
        self.league_commitments = []
        self.load_league_commitments()
        self.models = self.load_huggingface_model()

    def load_huggingface_model(self):
        try:
            bt.logging.info("Loading models from Hugging Face")
            
            # Download and load multiple files
            model_keras = hf_hub_download(repo_id="sportstensor/basic_model", filename="model.keras")
            epl_model = hf_hub_download(repo_id="sportstensor/basic_model", filename="epl/model.keras")
            mlb_model = hf_hub_download(repo_id="sportstensor/basic_model", filename="mlb/model.keras")
            mls_model = hf_hub_download(repo_id="sportstensor/basic_model", filename="mls/model.keras")
            nfl_model = hf_hub_download(repo_id="sportstensor/basic_model", filename="nfl/model.keras")
            
            # Load the models as needed
            model = load_model(model_keras)
            epl_model = load_model(epl_model)
            mlb_model = load_model(mlb_model)
            mls_model = load_model(mls_model)
            nfl_model = load_model(nfl_model)
            
            bt.logging.info("Models loaded successfully")
            
            # Store models in a dictionary for easy access
            self.models = {
                "general": model,
                "epl": epl_model,
                "mlb": mlb_model,
                "mls": mls_model,
                "nfl": nfl_model
            }
            
            return self.models
        except Exception as e:
            bt.logging.error(f"Error loading models: {str(e)}")
            return None

    def load_league_commitments(self):
        league_commitments = os.getenv("LEAGUE_COMMITMENTS")
        leagues_list = league_commitments.split(",")

        leagues = []
        for league_string in leagues_list:
            try:
                league = get_league_from_string(league_string.strip())
                leagues.append(league)
            except ValueError:
                print(f"Warning: Ignoring invalid league '{league_string}'")

        if not leagues or len(leagues) == 0:
            bt.logging.error("No leagues found in the environment variable LEAGUE_COMMITMENTS.")
            self.league_commitments = []
        else:
            self.league_commitments = leagues

    async def get_match_prediction(self, synapse: GetMatchPrediction) -> GetMatchPrediction:
        bt.logging.info(
            f"Received GetMatchPrediction request in forward() from {synapse.dendrite.hotkey}."
        )

        # Make the match prediction based on the requested MatchPrediction object
        synapse.match_prediction = make_match_prediction(synapse.match_prediction)
        synapse.version = constants.PROTOCOL_VERSION

        bt.logging.success(
            f"Returning MatchPrediction to {synapse.dendrite.hotkey}: \n{synapse.match_prediction.pretty_print()}."
        )

        return synapse

    async def get_league_commitments(self, synapse: GetLeagueCommitments) -> GetLeagueCommitments:
        bt.logging.info(
            f"Received GetLeagueCommitments request in forward() from {synapse.dendrite.hotkey}."
        )
        
        # Load our league commitments from the environment variable every time we receive a request. Avoids miners having to restart
        self.load_league_commitments()
        synapse.leagues = self.league_commitments
        synapse.version = constants.PROTOCOL_VERSION

        bt.logging.success(
            f"Returning Leagues to {synapse.dendrite.hotkey}: {[league.value for league in synapse.leagues]}."
        )

        return synapse
    
    async def get_league_commitments_blacklist(
        self, synapse: GetLeagueCommitments
    ) -> typing.Tuple[bool, str]:
        return await self.blacklist(synapse)
    
    async def get_match_prediction_blacklist(
        self, synapse: GetMatchPrediction
    ) -> typing.Tuple[bool, str]:
        return await self.blacklist(synapse)

    async def blacklist(self, synapse: bt.Synapse) -> typing.Tuple[bool, str]:
        if not synapse.dendrite.hotkey:
            return True, "Hotkey not provided"
        registered = synapse.dendrite.hotkey in self.metagraph.hotkeys
        if self.config.blacklist.allow_non_registered and not registered:
            return False, "Allowing un-registered hotkey"
        elif not registered:
            bt.logging.trace(
                f"Blacklisting un-registered hotkey {synapse.dendrite.hotkey}"
            )
            return True, f"Unrecognized hotkey {synapse.dendrite.hotkey}"

        uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        if self.config.blacklist.force_validator_permit:
            if not self.metagraph.validator_permit[uid]:
                bt.logging.warning(
                    f"Blacklisting a request from non-validator hotkey {synapse.dendrite.hotkey}"
                )
                return True, "Non-validator hotkey"

        stake = self.metagraph.S[uid].item()
        if (
            self.config.blacklist.validator_min_stake
            and stake < self.config.blacklist.validator_min_stake
        ):
            bt.logging.warning(
                f"Blacklisting request from {synapse.dendrite.hotkey} [uid={uid}], not enough stake -- {stake}"
            )
            return True, "Stake below minimum"

        bt.logging.trace(
            f"Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}"
        )
        return False, "Hotkey recognized!"

    
    async def get_league_commitments_priority(self, synapse: GetLeagueCommitments) -> float:
        return await self.priority(synapse)
    
    async def get_match_prediction_priority(self, synapse: GetMatchPrediction) -> float:
        return await self.priority(synapse)

    async def priority(self, synapse: bt.Synapse) -> float:
        caller_uid = self.metagraph.hotkeys.index(
            synapse.dendrite.hotkey
        )  # Get the caller index.
        priority = float(
            self.metagraph.S[caller_uid]
        )  # Return the stake as the priority.
        bt.logging.trace(
            f"Prioritizing {synapse.dendrite.hotkey} with value: ", priority
        )
        return priority

    def save_state(self):
        pass

# This is the main function, which runs the miner.
if __name__ == "__main__":
    with Miner() as miner:
        while True:
            bt.logging.info(f"Sportstensor Miner running, committed to leagues: {[league.value for league in miner.league_commitments]}")
            time.sleep(60)
