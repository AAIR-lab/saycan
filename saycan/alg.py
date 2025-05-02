import saycan
from saycan.env import PickPlaceEnv
from saycan import prompt
from saycan import llm
from saycan import vild
from saycan import constants
from saycan import helper
import numpy as np
import time
import os
import tensorflow.compat.v1 as tf
import tempfile

class SayCan:

    SAVED_MODEL_DIRPATH = "%s/image_path_v2" % (saycan.ASSETS_DIR)

    def __init__(self, env, model_dirpath=SAVED_MODEL_DIRPATH):

        self.env = env

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
        self.session = tf.Session(graph=tf.Graph(), config=tf.ConfigProto(
            gpu_options=gpu_options))
        tf.saved_model.loader.load(self.session, ["serve"], model_dirpath)

        self.actions = llm.generate_possible_actions(constants.PICK_TARGETS,
            constants.PLACE_TARGETS,
            termination_string=prompt.TERMINATION_STRING)

        self.optimizer, num_params = vild.get_vild_optimizer()
        self.scene_description = None
        self.found_objects = None
        self.affordance_scores = None

        if saycan.VERBOSITY:
            print("SayCan is considering", len(self.actions), "actions")
            print(f'ViLD parameters: {num_params:,}')
    
    def initialize_vild_and_compute_affordances(self, output_dir):

        if self.found_objects is not None:
            return

        image_path = "%s/vild_object_detect.png" % (output_dir)
        helper.save_env_top_image(self.env, image_path, show=False)

        self.found_objects, scores = vild.vild(self.session, image_path, 
            constants.category_name_string, constants.vild_params,
            plot_on=False)

        if saycan.VERBOSITY >= 2:
            print("SayCan: Object detection on init state")
            for found_object, score in zip(self.found_objects, scores):
                print("Found a", found_object, "with score:", score)

        self.scene_description = helper.build_scene_description(
            self.found_objects)

        if saycan.VERBOSITY >= 2:
            print("SayCan scene description")
            print(self.scene_description)

        self.affordance_scores = helper.affordance_scoring(self.actions,
            self.found_objects, block_name="box", bowl_name="circle",
            verbose=False)

    def saycan(self, output_dir, task_prompt, client, state=None,
        max_horizon=10):
        
        os.makedirs(output_dir, exist_ok=True)
        self.initialize_vild_and_compute_affordances(output_dir)

        if state is None:
            self.env.set_state(self.env.init_state_id)

        image_path = "%s/saycan_init_s.png" % (output_dir)
        helper.save_env_top_image(self.env, image_path, show=False)

        high_level_actions = []
        selected_action = ""
        h = 0
        while selected_action != task_prompt.termination_string \
            and h < max_horizon:

            with open("%s/prompt-%u.txt" % (output_dir, h), "w") as fh:
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

            fig_filepath = "%s/%s.png" % (output_dir, selected_action)
            helper.plot(llm_scores, self.affordance_scores, saycan_scores,
                selected_action, fig_filepath=fig_filepath)

        return high_level_actions

    def execute_plan(self, output_dir, plan,
        state=None, store_video=True):

        os.makedirs(output_dir, exist_ok=True)

        with open("%s/plan.txt" % (output_dir), "w") as fh:
            fh.write("\n".join(plan))

        if state is None:
            self.env.set_state(self.env.init_state_id)
        
        steps = []
        for i, action in enumerate(plan):

            if action == prompt.TERMINATION_STRING:
                break

            steps.append(i)
            step_dir = "%s/step-%u_%s" % (output_dir, i, action)
            os.makedirs(step_dir, exist_ok=True)
            
            obs = self.env.get_state()
            nlp_action = helper.convert_action_to_cliport_action(action)
            vild.run_cliport(step_dir, self.optimizer, self.env, obs,
                nlp_action)

            if store_video:
                step_video_filepath = "%s/step_execution.mp4" % (step_dir)
                self.env.save_video(step_video_filepath, [i])

        if store_video:
            execution_filepath = "%s/plan_execution.mp4" % (output_dir)
            self.env.save_video(execution_filepath,  range(len(steps)))

if __name__ == "__main__":

    import shutil

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
    saycan_obj = SayCan(env)

    import openai
    api_key = "invalid"
    base_url = "http://en4230849l.scai.dhcp.asu.edu:8000/v1"
    MODEL_NAME = "meta-llama/Llama-3.1-8B"
    OUTPUT_DIR = "/tmp/results"
    CLEAN = True

    if CLEAN:
        shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    from saycan.llm import SayCanLLMClient
    from saycan.prompt import Prompt

    client = SayCanLLMClient(MODEL_NAME, api_key=api_key,
        base_url=base_url)

    task_prompt = Prompt(TASK)
    plan = saycan_obj.saycan(OUTPUT_DIR, task_prompt, client)

    # DUMMY_PLAN = ["robot.pick_and_place(green block, middle)",
    #     "robot.pick_and_place(yellow block, green block)",
    #     "robot.pick_and_place(red block, yellow block)",
    #     "done()"
    # ]
    # plan = DUMMY_PLAN

    saycan_obj.execute_plan(OUTPUT_DIR, plan, "/tmp/results")

