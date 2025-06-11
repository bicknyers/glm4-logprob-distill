import os

from litellm.integrations.custom_logger import CustomLogger
import litellm
from litellm.proxy.proxy_server import UserAPIKeyAuth, DualCache
from typing import Optional, Literal

from langfuse import Langfuse
from langfuse.decorators import observe

from litellm.types.utils import TopLogprob

LANGFUSE_HOST = os.environ.get("LANGFUSE_HOST")
LANGFUSE_PUBLIC_KEY = os.environ.get("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.environ.get("LANGFUSE_SECRET_KEY")

langfuse = Langfuse(host=LANGFUSE_HOST, public_key=LANGFUSE_PUBLIC_KEY, secret_key=LANGFUSE_SECRET_KEY)


# This file includes the custom callbacks for LiteLLM Proxy
# Once defined, these can be passed in proxy_config.yaml
class MyCustomHandler(CustomLogger): # https://docs.litellm.ai/docs/observability/custom_callback#callback-class
    # Class variables or attributes
    def __init__(self):
        pass

    #### CALL HOOKS - proxy only ####
    async def async_pre_call_hook(self, user_api_key_dict: UserAPIKeyAuth, cache: DualCache, data: dict, call_type: Literal[
            "completion",
            "text_completion",
            "embeddings",
            "image_generation",
            "moderation",
            "audio_transcription",
        ]):
        data["logprobs"] = True
        data["top_logprobs"] = 8
        data["top_p"] = 0.0001
        return data

    @observe()
    async def async_custom_logprob_hook(
        self,
        model,
        prompt,
        message,
        logprobs,
    ):
        pass


    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response,
    ):
        parsed_logprobs = []
        for logprob in response.choices[0].logprobs.content:
            temp_dict = {}
            temp_dict["tk"] = logprob['token']
            temp_dict["bt"] = logprob['bytes']
            temp_dict["lp"] = logprob['logprob']
            temp_top = []
            for top in logprob['top_logprobs']:
                temp_top_dict = {}
                temp_top_dict["tk"] = top.token
                temp_top_dict["bt"] = top.bytes
                temp_top_dict["lp"] = top.logprob
                temp_top.append(temp_top_dict)
            temp_dict["tp"] = temp_top
            parsed_logprobs.append(temp_dict)
        await self.async_custom_logprob_hook(data['model'], data['messages'], response.choices[0].message.content, parsed_logprobs)
        pass


proxy_handler_instance = MyCustomHandler()
