pip install tensorflow-gpu==2.0.0b1  -i https://pypi.tuna.tsinghua.edu.cn/simple

解决windows安装TensorFlow2.0beta版本时ERROR: Cannot uninstall 'wrapt'问题

https://www.cnblogs.com/xiaowei2092/p/11025155.html

pip install -U --ignore-installed wrapt enum34 simplejson netaddr


https://tensorflow.google.cn/api_docs/python/tf/saved_model/simple_save


save_model 
https://www.deeplearn.me/2493.html
https://blog.csdn.net/mogoweb/article/details/83021524


________________________________________________________________________________________

RemoveError: 'setuptools' is a dependency of conda and cannot be removed from
conda's operating environment.

conda install -c anaconda setuptools

________________________________________________________________________________________

tools for analyze the savedmodel 
just run the 'saved_model_cli' command
saved_model_cli show --dir model/1573126315/ --tag_set serve --signature_def serving_default
saved_model_cli show --dir model/1573126315/ --all
