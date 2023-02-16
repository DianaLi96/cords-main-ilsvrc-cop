# Learning setting
config = dict(setting="SL",
              is_reg = False,
              dataset=dict(name="cifar10",
                           datadir="/root/desc/datasets/cifar10",
                           feature="dss",
                           type="image"),

              dataloader=dict(shuffle=True,
                              batch_size=128,
                              pin_memory=True),

              model=dict(architecture='ResNet18',
                         type='pre-defined',
                         numclasses=10),

              ckpt=dict(is_load=False,
                        is_save=True,
                        dir='results/',
                        save_every=20),

              loss=dict(type='CrossEntropyLoss',
                        use_sigmoid=False),

              optimizer=dict(type="sgd",
                             momentum=0.9,
                             lr=0.1,
                             weight_decay=5e-4,
                             nesterov=True),

              scheduler=dict(type="multi_step"),

              dss_args=dict(type="GradMatchPB-Warm",
                            fraction=0.6,
                            select_every=10,
                            lam=0,
                            selection_type='PerBatch',
                            v1=True,
                            valid=False,
                            eps=1e-100,
                            linear_layer=True,
                            kappa=0.05),


              train_args=dict(num_epochs=200,
                              device="cuda",
                              print_every=10,
                              results_dir='results/',
                              print_args=["tst_loss", "tst_acc", "time"],
                              return_args=[]
                              )
              )
