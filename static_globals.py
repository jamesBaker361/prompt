BLIP_DIFFUSION="blip_diffusion"
ELITE="elite"
RIVAL="rival"
IP_ADAPTER="adapter"
FACE_IP_ADAPTER="face_adapter"
CHOSEN="chosen"

METHOD_LIST=[BLIP_DIFFUSION, ELITE, RIVAL,IP_ADAPTER,FACE_IP_ADAPTER,CHOSEN]

NEGATIVE="over-exposure, under-exposure, saturated, duplicate, out of frame, lowres, cropped, worst quality, low quality, jpeg artifacts, morbid, mutilated, out of frame, ugly, bad anatomy, bad proportions, deformed, blurry, duplicate"

#metrics:
PROMPT_SIMILARITY="prompt_similarity"
IDENTITY_CONSISTENCY="identity_consistency"
TARGET_SIMILARITY="target_similarity"
AESTHETIC_SCORE="aesthetic_score"
IMAGE_REWARD="image_reward"

METRIC_LIST=[PROMPT_SIMILARITY, IDENTITY_CONSISTENCY, TARGET_SIMILARITY, AESTHETIC_SCORE, IMAGE_REWARD]

TEXT_INPUT_IDS="text_input_ids"
CLIP_IMAGES='clip_images'
IMAGES="images" #in text_to_image_lora this is aka pixel_values