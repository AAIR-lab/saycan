import openai
from constants import *
import clip
from transformers import AutoTokenizer

openai.api_key = ""
#@title LLM Cache
overwrite_cache = True
if overwrite_cache:
  LLM_CACHE = {}
ENGINE = "text-davinci-003"
ENGINE = "text-ada-001"

class SayCanLLMClient:

  def __init__(self, model, temperature=0,
    api_key=None, base_url=None):

    self.model = model
    self.temperature = temperature
    self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
    self.cached_responses = {}
    self.tokenizer = AutoTokenizer.from_pretrained(model)

  def get_responses(self, prompts):

    assert isinstance(prompts, list)
    prompt_cache_key = "".join(prompts)

    try:
      response = self.cached_responses[prompt_cache_key]
    except KeyError:

      response = self.client.completions.create(model=self.model,
                                          prompt=prompts,
                                          max_tokens=0,
                                          temperature=self.temperature,
                                          logprobs=1,
                                          echo=True)

      # Convert it to the dictionary to match legacy code
      # openai new v1 responses use response.choices
      response = response.to_dict()
    
      self.cached_responses[prompt_cache_key] = response

    return response

  def get_scores(self, prompt, actions):

    few_shot_prompt = str(prompt)

    # The last token is the one we want to wait on since it marks
    # the start of all options
    assert few_shot_prompt[-1] == prompt.separator

    action_start_token = self.tokenizer.tokenize(few_shot_prompt)[-1]

    prompts = [few_shot_prompt + action for action in actions]
    response = self.get_responses(prompts)

    scores = {}
    for action, choice in zip(actions, response["choices"]):
      tokens = reversed(choice["logprobs"]["tokens"])
      token_logprobs = reversed(choice["logprobs"]["token_logprobs"])

      total_logprob = 0
      debug_end_found = False
      for token, token_logprob in zip(tokens, token_logprobs):
        if token == action_start_token:
          debug_end_found = True
          break
        total_logprob += token_logprob
      
      # Need to make sure that the right option was found and that the
      # whole prompt was not considered
      assert debug_end_found
      scores[action] = total_logprob

    return scores

def generate_possible_actions(pick_targets, place_targets, termination_string,
  actions_in_api_form=True):

  actions = []
  for pick in pick_targets:
    for place in place_targets:
      if actions_in_api_form:
        action = "robot.pick_and_place({}, {})".format(pick, place)
      else:
        action = "Pick the {} and place it on the {}.".format(pick, place)
      actions.append(action)

  actions.append(termination_string)
  return actions