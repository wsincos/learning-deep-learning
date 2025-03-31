# lightning

在设置种子的时候，需要设置很多seed，包括如下内容

```python
import random
import numpy as np
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)            # CPU 上设置随机种子
torch.cuda.manual_seed(SEED)       # GPU 上设置随机种子
torch.cuda.manual_seed_all(SEED)   # 多 GPU 时所有 GPU 的随机种子

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

```

而使用pytorch_lighting只需要一行

```python
from pytorch_lightning import seed_everything

seed = 42
seed_everything(seed, workers=True)

# 下面这些与seed无关，如果希望能够复现结果，就需要这个
# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

`pl.LightningModule` 

pl.LightningModule继承于nn.Module, 包含如下五个部分：

1. Initialization (`__init__`), where we create all necessary parameters/models
2. Optimizers (`configure_optimizers`) where we create the optimizers, learning rate scheduler, etc.
3. Training loop (`training_step`) where we only have to define the loss calculation for a single batch (the loop of optimizer.zero_grad(), loss.backward() and optimizer.step(), as well as any logging/saving operation, is done in the background)
4. Validation loop (`validation_step`) where similarly to the training, we only have to define what should happen per step
5. Test loop (`test_step`) which is the same as validation, only on a test set.
6. `forward` function

一个例子如下

```python

def create_model(model_name, model_hparams):
    if model_name in model_dict:
        return model_dict[model_name](**model_hparams)
    else:
        assert False, f'Unknown model name "{model_name}". Available models are: {str(model_dict.keys())}'

class CIFARModule(pl.LightningModule):
    def __init__(self, model_name, model_hparams, optimizer_name, optimizer_hparams):
        """CIFARModule.

        Args:
            model_name: Name of the model/CNN to run. Used for creating the model (see function below)
            model_hparams: Hyperparameters for the model, as dictionary.
            optimizer_name: Name of the optimizer to use. Currently supported: Adam, SGD
            optimizer_hparams: Hyperparameters for the optimizer, as dictionary. This includes learning rate, weight decay, etc.

        """
        # init操作内容：
        # 1.保存超参数，save_hyperparameters，这样还会给你创建一个self.hparams
        # 2.创建模型（可以提前先用nn.Module创建一个class）和loss
        # 注意，optimizers不在这里定义，而是有一个专门的函数
        # 3.其余操作
        super().__init__()
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters()
        # Create model
        self.model = create_model(model_name, model_hparams)
        # Create loss module
        self.loss_module = nn.CrossEntropyLoss()
        # Example input for visualizing the graph in Tensorboard
        self.example_input_array = torch.zeros((1, 3, 32, 32), dtype=torch.float32)

    def forward(self, imgs):
        return self.model(imgs)

    def configure_optimizers(self):
        # 在optimizers的定义中，我们需要考虑定义如下内容
        # 1.optimizer
        # 2.lr_scheduler（可以没有），在定义的时候，尽量用ReduceLROnPlateau
        # 注意，在返回的时候，尽量以列表的形式返回这两个东西，针对有多个optimizer的时候很有用
        if self.hparams.optimizer_name == "Adam":
            # AdamW is Adam with a correct implementation of weight decay (see here
            # for details: https://arxiv.org/abs/1711.05101)
            optimizer = optim.AdamW(self.parameters(), **self.hparams.optimizer_hparams)
        elif self.hparams.optimizer_name == "SGD":
            optimizer = optim.SGD(self.parameters(), **self.hparams.optimizer_hparams)
        else:
            assert False, f'Unknown optimizer: "{self.hparams.optimizer_name}"'

        # We will reduce the learning rate by 0.1 after 100 and 150 epochs
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
        # 更常用的方式如下
        # 这个不需要指定什么时候降低学习率，它会自己根据看什么时候改learning rate
        scheduler = {
		        "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
		            optimizer,
		            mode="min",              # 监控的指标越小越好（通常是 val_loss）
		            factor=0.1,              # 学习率缩小为原来的 0.1 倍
		            patience=10,            # 连续10个epoch无提升就调整
		            verbose=True            # 打印学习率
		        ),
		        "monitor": "val_loss",      # 关键：一定要self.log("val_loss")
		        "interval": "epoch",        # 每个 epoch 检查一次
		        "frequency": 1
		    }
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        # 注意training_step需要return，其余step不需要
        imgs, labels = batch
        preds = self.model(imgs)
        loss = self.loss_module(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log("train_acc", acc, on_step=False, on_epoch=True)
        self.log("train_loss", loss)
        return loss  # Return tensor to call ".backward" on

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs)
        loss = self.loss_module(preds, labels)
        preds = self.model(imgs).argmax(dim=-1)
        acc = (preds.argmax(dim=-1) == labels).float().mean()
        # By default logs it per epoch (weighted average over batches)
        self.log("val_acc", acc)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        

    def test_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs).argmax(dim=-1)
        acc = (labels == preds).float().mean()
        # By default logs it per epoch (weighted average over batches), and returns it afterwards
        self.log("test_acc", acc)
```

```python
# Callbacks
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

model_dict = {}

def create_model(model_name, model_hparams):
    if model_name in model_dict:
        return model_dict[model_name](**model_hparams)
    else:
        assert False, f'Unknown model name "{model_name}". Available models are: {str(model_dict.keys())}'
        
act_fn_by_name = {"tanh": nn.Tanh, "relu": nn.ReLU, "leakyrelu": nn.LeakyReLU, "gelu": nn.GELU}

def train_model(model_name, save_name=None, **kwargs):
    """Train model.

    Args:
        model_name: Name of the model you want to run. Is used to look up the class in "model_dict"
        save_name (optional): If specified, this name will be used for creating the checkpoint and logging directory.

    """
    if save_name is None:
        save_name = model_name

    # Create a PyTorch Lightning trainer with the generation callback
    trainer = pl.Trainer(
        default_root_dir=os.path.join(CHECKPOINT_PATH, save_name),
        accelerator="auto",
        devices=1,
        max_epochs=180,
        callbacks=[
            ModelCheckpoint(
                save_weights_only=True, mode="max", monitor="val_acc"
            ),
            LearningRateMonitor("epoch"),
        ],
    )
    trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, save_name + ".ckpt")
    if os.path.isfile(pretrained_filename):
        print(f"Found pretrained model at {pretrained_filename}, loading...")
        # Automatically loads the model with the saved hyperparameters
        model = CIFARModule.load_from_checkpoint(pretrained_filename)
    else:
        pl.seed_everything(42)  # To be reproducible
        model = CIFARModule(model_name=model_name, **kwargs)
        trainer.fit(model, train_loader, val_loader)
        model = CIFARModule.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path
        )  # Load best checkpoint after training

    # Test best model on validation and test set
    val_result = trainer.test(model, dataloaders=val_loader, verbose=False)
    test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
    result = {"test": test_result[0]["test_acc"], "val": val_result[0]["test_acc"]}

    return model, result
```

```python
resnet_model,resnet_results=train_model(
		model_name="ResNet",
		model_hparams={"num_classes": 10, 
									 "c_hidden":[16, 32, 64], 
									 "num_blocks":[3, 3, 3], 
									 "act_fn_name": "relu"
									 },
		optimizer_name="SGD",
		optimizer_hparams={"lr": 0.1, "momentum": 0.9, "weight_decay": 1e-4},
		)
```

使用tensorboard

```python
from pytorch_lightning.loggers import TensorBoardLogger
logger = TensorBoardLogger("logs", name="my_model")  # logs/my_model/
```

在trainer中：

```python
trainer = pl.Trainer(
			logger=logger,
			max_epochs=50,
			# 可选：用于显示模型图结构
			enable_model_summary=True
			)
```

在模型中；

```python
def training_step(self, batch, batch_idx):
    x, y = batch
    logits = self(x)
    loss = F.cross_entropy(logits, y)

    # 记录训练 loss
    self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    return loss
```

其余可选项：

trainer定义以后，`trainer.logger._log_graph = True` 可以绘制模型结构图

可以利用这个函数，在模型训练后，打印出来

```python
def validation_epoch_end(self, outputs):
    accs = [x["valid/acc"] for x in outputs if "valid/acc" in x]
    avg_acc = torch.stack(accs).mean()
    print(f"Epoch {self.current_epoch} - Valid Acc: {avg_acc:.4f}")

```

完整流程如下：

- 自己定义好模型和train/valid/test dataloader
- 编写pl.LightningModule
    - 首先编写init的时候，有几个东西是规范的：
        
        `model_name` `model_hparams`  `optimizer_name` `optimizer_hparams` 
        
        也就是说，你只需要挑选好你的模型参数、optimizer的参数就行了
        
        在写init的时候，首先第一步，需要讲这些参数以`self.save_hyperparameters()` 的方式进行保存，方便后续直接利用self.hparams.xxx进行利用
        
        然后需要设定好`model`和`loss`
        
    - 写完init，简单写完forward
    - configure_optimizers
        
        首先你要确定自己的optimizer是什么，定义optimizer(lr和weight decay)
        
        <aside>
        💡
        
        由于在常规的方式中，model、loss、optimizer都是一起写的，而在这里optimizer却另外有一个函数来写了，所以要记住这种差异
        
        </aside>
        
        然后写scheduler，注意这里建议使用`ReduceLROnPlateau`，想好要在scheduler中需要哪些东西：建议的范式如下
        
        ```python
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                mode='min',
                factor=0.1,
                patience=10,
                verbose=True,
            ),
            "monitor": "valid/loss",
            "interval": "epoch",
            "frequency": 1
        }
        return [optimizer], [scheduler]
        ```
        
    - 写几个step, 只需要注意开头要写上batch 和 batch idx
    - 如果每个epoch想要有输出，还可以用on_validation_epoch_end，结合`self.trainer.callback_metrics.get("valid/acc")`来打印一下epoch的结果

- 写train函数
    - 写trainer
        
        写trainer的时候，就是从大局观的角度来编写模型，那么就应该要想如下的东西
        
        **地址（可以没有）**、**是否使用加速**，**如果用的话，用哪几个**、**模型训练几轮**
        
        以及每轮epoch以后，是否使用callbacks（保存模型和learning rate）
        
        <aside>
        💡
        
        callbacks非常重要
        
        </aside>
        
        还有就是设置logger的两个细枝末节的东西：`logger._log_graph`和`logger.__default_hp_metrics` 
        
    - 写好trainer以后，就需要用fit函数将train和valid dataloader传入进去了
    - 模型训练完成以后，调用Lightning Module的`load_from_checkpoint(trainer.checkpoint_callback.best_model_path)`
        
        函数，调用这个模型，得到最好的模型