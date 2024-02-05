from transformers import Trainer, TrainingArguments
from datasets import load_dataset
from transformers import ViTFeatureExtractor
from transformers import ViTForImageClassification
from sklearn.metrics import confusion_matrix

from datasets import load_metric
import numpy as np
iterate_cm = 0

metric = load_metric("accuracy")
def compute_metrics(p):
    global iterate_cm
    y_true = p.label_ids
    y_pred = p.predictions.argmax(-1)
    confusion_matrix_result = confusion_matrix(y_true, y_pred)
    np.save(f"./drive/MyDrive/cifar100/cm/confusion_matrix_{iterate_cm}",confusion_matrix_result)
    iterate_cm += 1
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)
### load cifar-100 dataset ###
ds = load_dataset('cifar100')
ds = ds.rename_column("fine_label", "labels")

N_epochs = 10
training_args_dict = {
    "output_dir": "./drive/MyDrive/cifar100",
    "per_device_train_batch_size": 80,
    "per_device_eval_batch_size": 80,
    "evaluation_strategy": "steps",
    "num_train_epochs": N_epochs,
    "fp16": True,
    "save_steps": 100,
    "eval_steps": 100,
    "logging_steps": 10,
    "learning_rate": 2e-4,
    "save_total_limit": 2,
    "remove_unused_columns": False,
    "push_to_hub": False,
    "report_to": 'tensorboard',
    "load_best_model_at_end": True,
    }


model_name_or_path = 'google/vit-base-patch16-224-in21k'
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name_or_path)

def transform(example_batch):

    # Take a list of PIL images and turn them to pixel values
    example_batch['image'] = [np.array(x) for x in example_batch['img']]
    example_batch['image'] = [x if len(x.shape) == 3 else np.stack([x, x, x], axis=-1) for x in example_batch['image']]
    ### Add noise of variance sigma^2 ###
    inputs = feature_extractor([x for x in example_batch['image']], return_tensors='pt')
    # Don't forget to include the labels!
    inputs['labels'] = example_batch['labels']
    return inputs
labels = ds['train'].features['labels'].names

prepared_ds = ds.with_transform(transform) 


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
    eval_dataset=prepared_ds['test'],
    compute_metrics=compute_metrics
)

trainer.train()
trainer.evaluate()
trainer.save_model("vit-base-cifar100")

#### Compute confusion matrix for each epoch ####
steps = trainer.eval_steps 

for step in steps:
    trainer.load_model(f"./drive/MyDrive/cifar100/checkpoint-{step}")
    preds = trainer.predict(prepared_ds['valid'])
    y_true = preds.label_ids
    y_pred = preds.predictions.argmax(-1)
    confusion_matrix_result = confusion_matrix(y_true, y_pred)
    np.save(f"confusion_matrix_step_{step}",confusion_matrix_result)



