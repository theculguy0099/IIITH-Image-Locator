- Used transfer learning to train a model to predict the region ID of a given image.
- Added few layers to the pre-trained model to adapt it to the new task.
- Data augmentation is done.
- efficientnet_b0, resnet50, convnext_tiny, convnext_base, vit_base_patch16_224 are used as base models. The predictions are then ensembled using average(or max).
- Run for 200 epochs on mini batch.

https://iiithydresearch-my.sharepoint.com/:f:/g/personal/priet_ukani_research_iiit_ac_in/EoDwYcwvdlRPm_NNLXsq9SQBkXnhW3z4YbOusv2Ix5sw1Q?e=eUKogh