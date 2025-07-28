- Ensemble models over different backbones.
- Main backbone model: Dino V2 from Facebook Research.
- CSVs were generated from each model and then ensembled.
- The ensemble was done using the average of the predictions from each model.
- Have used the predicted regionIDs from the region Model and used it to train 15 different models for each region to predict the latlong. 
- Huber L1 smooth loss is used for penalising the outlier values. 
- Data augmentation and other some techniques are used to improve the performance of the model.

https://iiithydresearch-my.sharepoint.com/:f:/g/personal/priet_ukani_research_iiit_ac_in/EoDwYcwvdlRPm_NNLXsq9SQBkXnhW3z4YbOusv2Ix5sw1Q?e=eUKogh