import copy

clip_env_prompts = {
    "metaworld_sweep-into-v2": "The green cube is in the hole.", # unsolved there is reward issue
    "metaworld_drawer-open-v2": "The drawer is opened.", # let's try the flipped version.
    "metaworld_door-open-v2": "The safe door is opened.", # let's try the flipped version.
    "metaworld_soccer-v2": "The soccer ball is in the goal.", # not solved, there is reward issue
    "metaworld_hammer-v3-goal-observable": "a robot gripper hammering a screw on the wall.",
    "metaworld_push-wall-v3-goal-observable": "a robot gripper bypassing a wall and pushing a cylinder.",
    "metaworld_faucet-close-v3-goal-observable": "a robot gripper rotating the faucet clockwise.",
    "metaworld_push-back-v3-goal-observable": "a robot gripper pushing the wooden block backward.",
    "metaworld_stick-pull-v3-goal-observable": "a robot gripper grasping a cuboid and pulling a kettle with the cuboid.",
    "metaworld_handle-press-side-v3-goal-observable": "a robot gripper pressing a handle down sideways.",
    "metaworld_push-v3-goal-observable": "a robot gripper pushing a cylinder.",
    "metaworld_shelf-place-v3-goal-observable": "a robot gripper picking and placing a cube onto a shelf.",
    "metaworld_window-close-v3-goal-observable": "a robot gripper pushing and closing a window.",
    "metaworld_peg-unplug-side-v3-goal-observable": "a robot gripper unpluging a peg sideways.",

    "CartPole-v1": "pole vertically upright on top of the cart.",
    
    "softgym_RopeFlattenEasy": "The blue rope is straightened.",
    "softgym_PassWater": "The container, which holds water, is as close to the red circle as possible without causing too many water droplets to spill.",
    "softgym_ClothFoldDiagonal": "The cloth is folded diagonally from top left corner to bottom right corner.",
}

# what RL-VLM-F uses
goal_env_prompts = {
    "metaworld_sweep-into-v2": "to minimize the distance between the green cube and the hole", # unsolved there is reward issue
    "metaworld_drawer-open-v2": "to open the drawer", # let's try the flipped version.
    "metaworld_door-open-v2": "to open the safe door", # let's try the flipped version.
    "metaworld_soccer-v2": "to move the soccer ball into the goal", # not solved, there is reward issue
    "metaworld_hammer-v3-goal-observable": "to hammer a screw on the wall",
    "metaworld_push-wall-v3-goal-observable": "to bypass the wall and push the cylinder",
    "metaworld_faucet-close-v3-goal-observable": "to rotate the faucet clockwise to close it",
    "metaworld_push-back-v3-goal-observable": "to push the wooden block backward to the target",
    "metaworld_stick-pull-v3-goal-observable": "to grasp the stick and pull the kettle to the target",
    "metaworld_handle-press-side-v3-goal-observable": "to press the side handle downward",
    "metaworld_push-v3-goal-observable": "to push the cylinder to the target",
    "metaworld_shelf-place-v3-goal-observable": "to pick and place the cube onto the shelf",
    "metaworld_window-close-v3-goal-observable": "to push and fully close the window",
    "metaworld_peg-unplug-side-v3-goal-observable": "to unplug the peg sideways from the socket",
    "CartPole-v1": "to balance the brown pole on the black cart to be upright",
    "softgym_RopeFlattenEasy": "to straighten the blue rope",
    "softgym_PassWater": "to move the container, which holds water, to be as close to the red circle as possible without causing too many water droplets to spill",
    "softgym_ClothFoldDiagonal": "to fold the cloth diagonally from top left corner to bottom right corner",
}


##########################################################################
### asking gemini to output a preference with 2 stage analysis ###############
#########################################################################
gemini_free_query_prompt1 =  """
Consider the following two images:
Image 1:
"""

gemini_free_query_prompt2 = """
Image 2:
"""

gemini_free_query_env_prompts = {}
gemini_free_query_template = """
1. What is shown in Image 1?
2. What is shown in Image 2?
3. The goal is {}. Is there any difference between Image 1 and Image 2 in terms of achieving the goal?
"""

for env_name, prompt in goal_env_prompts.items():
    gemini_free_query_env_prompts[env_name] = gemini_free_query_template.format(prompt)
    
### preference summary prompt
gemini_summary_env_prompts = {}

### template 1 
gemini_summary_template = """
Based on the text below to the questions:
1. What is shown in Image 1?
2. What is shown in Image 2?
3. The goal is {}. Is there any difference between Image 1 and Image 2 in terms of achieving the goal?
{}

Is the goal better achieved in Image 1 or Image 2?
Reply a single line of 0 if the goal is better achieved in Image 1, or 1 if it is better achieved in Image 2.
Reply -1 if the text is unsure or there is no difference.
""" 

for env_name, prompt in goal_env_prompts.items():
    gemini_summary_env_prompts[env_name] = gemini_summary_template.format(prompt, "{}")
    
######################################################################
### asking gemini to output a preference with 1 stage analysis ###############
######################################################################

gemini_single_query_prompt_template = """
1. What is shown in Image 1?
2. What is shown in Image 2?
3. The goal is {}. Is there any difference between Image 1 and Image 2 in terms of achieving the goal?

Is the goal better achieved in Image 1 or Image 2?
Reply a single line of 0 if the goal is better achieved in Image 1, or 1 if it is better achieved in Image 2.
Reply -1 if the text is unsure or there is no difference.
"""

gemini_single_query_env_prompts = {}
for env_name, prompt in goal_env_prompts.items():
    gemini_single_query_env_prompts[env_name] = gemini_single_query_prompt_template.format(prompt)
    
######################################################################
### asking gemini to output a preference with only labeling ###############
######################################################################
gemini_single_query_no_analysis_prompt_template = """
The goal is {}. Is the goal better achieved in Image 1 or Image 2?
At the end of the response, reply a single line of: 
0 if the goal is better achieved in Image 1, 
1 if it is better achieved in Image 2, or
-1 if there is no difference or if it is unclear.
"""

### asking gemini to output score
gemini_score_prompt_start = """
Consider the following image:
"""

gemini_score_template = """
1. What is shown in the image?
2. The goal is {}. On a scale of 0 to 1, the score is 1 if the goal is achieved. What score would you give the image in terms of achieving the goal?
"""

gemini_score_env_prompts = {}
for env_name, prompt in goal_env_prompts.items():
    gemini_score_env_prompts[env_name] = gemini_score_template.format(prompt)

gemini_score_summary_template = """
Based on the text below to the questions: 
1. What is shown in the image?
2. The goal is {}. On a scale of 0 to 1, the score is 1 if the goal is achieved. What score would you give the image in terms of achieving the goal?
{}

Please reply a single line of the score the text has given.
Reply -1 if the text is unsure.
"""

gemini_score_summary_env_prompts = {}
for env_name, prompt in goal_env_prompts.items():
    gemini_score_summary_env_prompts[env_name] = gemini_score_summary_template.format(prompt, "{}")
    
    
### GPT uses the same prompt template as gemini
gpt_free_query_env_prompts = {}
gpt_free_query_template = copy.deepcopy(gemini_free_query_template)
for env_name, prompt in goal_env_prompts.items():
    gpt_free_query_env_prompts[env_name] = gpt_free_query_template.format(prompt)

gpt_summary_env_prompts = {}
gpt_summary_template = copy.deepcopy(gemini_summary_template)
for env_name, prompt in goal_env_prompts.items():
    gpt_summary_env_prompts[env_name] = gpt_summary_template.format(prompt, "{}")


gpt_score_query_env_prompts = {}
gpt_score_template = copy.deepcopy(gemini_score_template)
for env_name, prompt in goal_env_prompts.items():
    gpt_score_query_env_prompts[env_name] = gpt_score_template.format(prompt)

gpt_score_summary_env_prompts = {}
gpt_score_summary_template = copy.deepcopy(gemini_score_summary_template)
for env_name, prompt in goal_env_prompts.items():
    gpt_score_summary_env_prompts[env_name] = gpt_score_summary_template.format(prompt, "{}")
