import saycan
from saycan.env import PickPlaceEnv
from saycan import prompt
from saycan import llm
from saycan import vild
from saycan import constants
from saycan import prompt
from saycan import helper
import numpy as np
import time
import shutil
import os
import tensorflow.compat.v1 as tf

class SayCan:

    SAVED_MODEL_DIRPATH = "%s/image_path_v2" % (saycan.ASSETS_DIR)

    def __init__(self, env, model_dirpath=SAVED_MODEL_DIRPATH,
        output_dir=saycan.OUTPUT_DIR, clean=True):

        if clean:
            shutil.rmtree(output_dir, ignore_errors=True)
        os.makedirs(output_dir, exist_ok=True)

        self.env = env
        self.output_dir = output_dir

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
        self.session = tf.Session(graph=tf.Graph(), config=tf.ConfigProto(
            gpu_options=gpu_options))
        tf.saved_model.loader.load(self.session, ["serve"], model_dirpath)

        self.actions = llm.generate_possible_actions(constants.PICK_TARGETS,
            constants.PLACE_TARGETS,
            termination_string=prompt.TERMINATION_STRING)

        if saycan.VERBOSITY >= 2:
            print("SayCan is considering", len(self.actions), "actions")

        self.env_description, self.found_objects = self._get_env_description()

        self.affordance_scores = helper.affordance_scoring(self.actions,
            self.found_objects, block_name="box", bowl_name="circle",
            verbose=False)

        self.task_no = 0
    
    def _get_env_description(self):

        image_path = "%s/env_init_image.png" % (self.output_dir)
        helper.save_env_top_image(self.env, image_path, show=False)

        found_objects, scores = vild.vild(self.session, image_path, 
            constants.category_name_string, constants.vild_params,
            plot_on=False)

        if saycan.VERBOSITY >= 2:
            print("SayCan: Object detection on init state")
            for found_object, score in zip(found_objects, scores):
                print("Found a", found_object, "with score:", score)

        scene_description = helper.build_scene_description(found_objects)

        if saycan.VERBOSITY >= 2:
            print("SayCan scene description")
            print(scene_description)

        return scene_description, found_objects

    def saycan(self, task_prompt, client, max_horizon=10):
        
        task_dir = "%s/task%u" % (self.output_dir, self.task_no)
        os.makedirs(task_dir, exist_ok=True)
        print("Saving data to:", task_dir)

        high_level_actions = []
        selected_action = ""
        h = 0
        while selected_action != task_prompt.termination_string \
            and h < max_horizon:

            with open("%s/prompt-%u.txt" % (task_dir, h), "w") as fh:
                fh.write(str(task_prompt))

            llm_scores = client.get_scores(task_prompt, self.actions)

            saycan_scores = {
                a: np.exp(llm_scores[a]) * self.affordance_scores[a]
                for a in self.actions
            }

            saycan_scores = helper.normalize_scores(saycan_scores)
            selected_action = max(saycan_scores, key=saycan_scores.get)
            high_level_actions.append(selected_action)

            task_prompt.append(selected_action, add_separator=True)
            h += 1

            fig_filepath = "%s/%s.png" % (task_dir, selected_action)
            helper.plot(llm_scores, self.affordance_scores, saycan_scores,
                selected_action, fig_filepath=fig_filepath)

        self.task_no += 1
        return high_level_actions

    def execute_plan(self, high_level_actions):

        pass


if __name__ == "__main__":

    print("SayCan Algorithm")

    SEED = None

    if SEED is None:
        SEED = int(time.time())

    print("Using seed:", SEED)
    np.random.seed(SEED)

    TASK = "put all the blocks in different corners"
    TASK_CONFIG = {
        "pick":  ["red block", "yellow block", "green block", "blue block"],
        "place": ["red bowl"]
    }

    # Show if JAX is using GPU.
    # from jax.lib import xla_bridge
    # print("Using CPU/GPU:", xla_bridge.get_backend().platform)

    env = PickPlaceEnv(TASK_CONFIG)
    _ = env.reset()
    saycan = SayCan(env, clean=False)

    import openai
    api_key = "invalid"
    base_url = "http://en4230849l.scai.dhcp.asu.edu:8000/v1"
    MODEL_NAME = "meta-llama/Llama-3.1-8B"

    from saycan.llm import SayCanLLMClient
    from saycan.prompt import Prompt

    client = SayCanLLMClient(MODEL_NAME, api_key=api_key,
        base_url=base_url)

    prompt = Prompt(TASK)
    saycan.saycan(prompt, client)

