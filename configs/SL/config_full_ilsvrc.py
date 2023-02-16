# Learning setting
config = dict(setting="SL",
              is_reg = False,
              dataset=dict(name="ilsvrc12",
                           datadir="/root/desc/dataset",
                           feature="dss",
                           type="image"),

              dataloader=dict(shuffle=True,
                              batch_size=512,
                              pin_memory=True),

              model=dict(architecture='ResNet18',
                         type='pre-defined',
                         numclasses=1000),
              
              ckpt=dict(is_load=False,
                        is_save=True,
                        dir='results_ilsvrc/',
                        save_every=20),
              
              loss=dict(type='CrossEntropyLoss',
                        use_sigmoid=False),

              optimizer=dict(type="sgd",
                             momentum=0.9,
                             lr=0.1,
                             weight_decay=1e-4,
                             nesterov=True),

              scheduler=dict(type="multi_step"),

              dss_args=dict(type="Full"),

              train_args=dict(num_epochs=120,
                              device="cuda",
                              print_every=1,
                              results_dir='results_ilsvrc/',
                              print_args=["tst_loss", "tst_acc", "time"],
                              return_args=[],
                              distributed=True,
                              rank=-1,
                              world_size=1,
                              dist_backend='nccl'
                              )
              )
