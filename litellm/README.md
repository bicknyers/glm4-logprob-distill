### Overview
[LiteLLM](https://github.com/BerriAI/litellm) requires some configuration, namely a script to pre-hook Roo Code requests to add logprob parameters, and a post-hook to extract the logprobs to hand off to Langfuse.

### Modifications
- Added Langfuse env. var. to docker-compose.yml to allow writing large files to Langfuse
- Force build of Dockerfile so we can use a version of the Langfuse SDK that supports these env. var.
- Add logprobs_hook.py in docker-compose.yml, litellm_config.yml, etc.
- Note the "compression" scheme used in the logprobs post-hook to be able to write longer messages into Langfuse
- If using OpenRouter, you will want to whitelist/blacklist certain model providers as some will claim to support logprobs, but they actually return garbage