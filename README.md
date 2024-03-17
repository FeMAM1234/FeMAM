This repository contains the implementation and supplementary materials for the research paper "Multi-Level Additive Modeling for Fine-grained Non-IID Federated Learning." 

Please first place the original cifar100 and tinyimagent dataset in ./dump_items dataset.
Then run the sh files in shs/generatedata to produce the data partition setting. For example, cifar10.sh means generate dirichlet distribution dataset with parameter 10. 
Finally run the corresponding sh file in shs/main to start training. For example, cifar10.sh means run dirichlet distribution with parameter 10. 
