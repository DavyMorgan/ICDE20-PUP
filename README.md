# PUP
This is the official implementation of our ICDE'20 and TKDE papers:  

Yu Zheng, Chen Gao, Xiangnan He, Yong Li, Depeng Jin, **Price-aware Recommendation with Graph Convolutional Networks**, In Proceedings of IEEE ICDE 2020.

Yu Zheng, Chen Gao, Xiangnan He, Yong Li, Depeng Jin, **Incorporating Price into Recommendation with Graph Convolutional Networks**, IEEE Transactions on Knowledge and Data Engineering.

***
First download the Yelp dataset ([link](https://www.yelp.com/dataset)) and the category file ([link](https://www.yelp.com/developers/documentation/v3/all_category_list/categories.json)).


Then generate training data and test data using the codes in src/yelp_restaurant_pipeline:
```
python generate_action_log.py
python generate_bridge.py
python generate_training_data_sample_fm_bpr.py
```


Then start the visdom server:
```
visdom -port 33332
```


Then simply run the following command to reproduce the experiments:
```
python app.py --flagfile ./config/yelp.cfg
```

If you use our codes and datasets in your research, please cite:
```
@inproceedings{zheng2020price,
  title={Price-aware recommendation with graph convolutional networks},
  author={Zheng, Yu and Gao, Chen and He, Xiangnan and Li, Yong and Jin, Depeng},
  booktitle={2020 IEEE 36th International Conference on Data Engineering (ICDE)},
  pages={133--144},
  year={2020},
  organization={IEEE}
}

@article{zheng2021incorporating,
  title={Incorporating price into recommendation with graph convolutional networks},
  author={Zheng, Yu and Gao, Chen and He, Xiangnan and Jin, Depeng and Li, Yong},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2021},
  publisher={IEEE}
}
```
