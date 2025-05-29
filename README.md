Logprob distillation of Deepseek-R1-0528 -> GLM-4 32B for Roo Code.

Will include customized GLM-4 training code, configs/code for LiteLLM + Langfuse for collecting the logprobs from OpenRouter, and other goodies.

Writeup is in the Wiki. Here are the (mostly) finished pages:
* [01.-Planning](https://github.com/bicknyers/glm4-logprob-distill/wiki/01.-Planning)

Contains modified code from GLM-4, some inspiration from DistillKit.

### Deployment Notes
All of my deployments were done locally using docker compose on Ubuntu LTS. Your mileage may vary.

[Langfuse](https://github.com/langfuse/langfuse) can be deployed using their instructions. I had connectivity issues with minio (for exporting logprobs data to .json) in my setup using the latest build, so I pinned to version 3.39 to circumvent this issue.

[LiteLLM](https://github.com/BerriAI/litellm) requires some configuration, namely a script to pre-hook Roo Code requests to add logprob parameters, and a post-hook to extract the logprobs to hand off to Langfuse. Check the litellm directory in this repo. for more info.
