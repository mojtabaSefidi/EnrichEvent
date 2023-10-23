from transformers import TrainingArguments, Trainer

class Fine_Tuning():

  def __init__(self,
              epochs,
              train_batchsize,
              batch_update,
              apex_opt_level,
              warmup_steps,
              learning_rate,
              epsilon):

    self.training_args = TrainingArguments(output_dir="/content/",
                                           num_train_epochs=epochs,
                                           per_device_train_batch_size=train_batchsize,
                                           per_device_eval_batch_size=train_batchsize,
                                           gradient_accumulation_steps=batch_update,
                                           evaluation_strategy="epoch",
                                           save_strategy = 'epoch',
                                           fp16=True,
                                           fp16_opt_level=apex_opt_level,
                                           warmup_steps=warmup_steps,
                                           learning_rate=learning_rate,
                                           adam_epsilon=epsilon,
                                           weight_decay=0.01,
                                           save_total_limit=1,
                                           load_best_model_at_end=True)

  def train(self,
            language_model,
            tokenizer,
            train_dataset):
    
    trainer = Trainer(model=language_model,
                      args=self.training_args,
                      train_dataset=train_dataset,
                      eval_dataset=train_dataset,
                      tokenizer=tokenizer)
    trainer.train()
    trainer.save_model()
    return