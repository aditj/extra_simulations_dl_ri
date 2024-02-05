from transformers import Trainer, TrainingArguments
from datasets import load_dataset
from transformers import ViTFeatureExtractor
from transformers import ViTForImageClassification
from sklearn.metrics import confusion_matrix

from datasets import load_metric
import numpy as np
metric = load_metric("accuracy")
def compute_metrics(p):
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)
ds = load_dataset('zh-plus/tiny-imagenet')
ds = ds.rename_column("label", "labels")

training_args_dict = {
    "output_dir": "./imagenet",
    "per_device_train_batch_size": 100,
    "per_device_eval_batch_size": 100,
    "evaluation_strategy": "steps",
    "num_train_epochs": 1,
    "fp16": True,
    "save_steps": 500,
    "eval_steps": 500,
    "logging_steps": 10,
    "learning_rate": 2e-4,
    "save_total_limit": 1,
    "remove_unused_columns": False,
    "push_to_hub": False,
    "report_to": 'tensorboard',
    "load_best_model_at_end": True,
    }


model_name_or_path = 'google/vit-base-patch16-224-in21k'
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name_or_path)

def transform(example_batch,sigma):

    # Take a list of PIL images and turn them to pixel values
    example_batch['image'] = [np.array(x) for x in example_batch['image']]
    example_batch['image'] = [x if len(x.shape) == 3 else np.stack([x, x, x], axis=-1) for x in example_batch['image']]

    ### Add noise of variance sigma^2 ###
    inputs = feature_extractor([x for x in example_batch['image']], return_tensors='pt')
    inputs['pixel_values'] = inputs['pixel_values'] + np.random.normal(0, sigma, inputs['pixel_values'].shape)
    # Don't forget to include the labels!
    inputs['labels'] = example_batch['labels']
    return inputs
labels = ds['train'].features['labels'].names

sigmas = [x for x in np.logspace(-3,0,10) if x not in [0.001,0.01, 0.1, 1]]

sigmas = [ 0.046415888336127774, 0.21544346900318823, 0.46415888336127775]

transforms = [lambda x: transform(x, sigma) for sigma in sigmas]


for j in range(len(sigmas)):
    sigma = sigmas[j]
    transform_j = transforms[j]
    prepared_ds = ds.with_transform(transform_j)
    model = ViTForImageClassification.from_pretrained(
        model_name_or_path,
        num_labels=len(labels),
        id2label={str(i): c for i, c in enumerate(labels)},
        label2id={c: str(i) for i, c in enumerate(labels)}
    )



    # Convert the dictionary to a TrainingArguments instance
    training_args = TrainingArguments(**training_args_dict)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=prepared_ds['train'],
        eval_dataset=prepared_ds['valid'],
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.evaluate()

    #### Compute confusion matrix ####

    preds = trainer.predict(prepared_ds['valid'])
    y_true = preds.label_ids
    y_pred = preds.predictions.argmax(-1)
    confusion_matrix_result = confusion_matrix(y_true, y_pred)
    np.save(f"./cm/variance/confusion_matrix_sigma_{sigma}",confusion_matrix_result)



