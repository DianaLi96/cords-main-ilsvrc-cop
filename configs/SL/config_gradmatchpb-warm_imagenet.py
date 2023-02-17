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
                        dir='results_ilsvrc12/',
                        save_every=10),

              loss=dict(type='CrossEntropyLoss',
                        use_sigmoid=False),

              optimizer=dict(type="sgd",
                             momentum=0.9,
                             lr=0.1,
                             weight_decay=1e-4,
                             nesterov=True),

              scheduler=dict(type="multi_step"),

              dss_args=dict(type="GradMatch-Warm",
                            fraction=0.6,
                            select_every=10,
                            lam=0,
                            selection_type='PerBatch',
                            v1=True,
                            valid=False,
                            eps=1e-100,
                            linear_layer=True,
                            kappa=0.0833), # 0.08333


              train_args=dict(num_epochs=120,
                              device="cuda",
                              print_every=1,
                              results_dir='results_ilsvrc12/',
                              print_args=["tst_loss", "tst_acc", "time"],
                              return_args=[],
                              distributed=True,
                              rank=-1,
                              world_size=1,
                              dist_backend='nccl'
                              )
              )

