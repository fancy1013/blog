```
titile: "Python：more than basic "
date: Jan 16, 2021
```

在看别人代码的过程中，遇到一些在一般的学习中不会遇到的技巧。

# 一、TODO comments

众所周知，python的注释语句用`#`或```。

在**编辑器中输入注释，注释部分的一些特殊的语句会被高亮**，比如`TODO`，用来表明要做的事情或者未完成的事情。

```python

print("something")

# TODO: finish the structure
```

在Pycharm中，可以直接对带`TODO`的注释统计，以此更方便进行代码完善。

这是[Pycharm官方说明。](https://www.jetbrains.com/help/pycharm/using-todo.html?gclid=CjwKCAiAl4WABhAJEiwATUnEF6UvndUf6vAqR4GbKnus-p7iyJ4caDVgiheeFG0Vr17sujKAkUIahxoCE-IQAvD_BwE)

还有一个类似的：`FIXME`，可以用来表明有bug需要修复。

细节虽小，意义重要。



# 二、Hydra

Hydra是一个开源Python框架，用于简化研究开发与一些复杂应用。

**应用一，自动加载yaml文件。**

示例代码：

config.yaml:

```yaml
db:
  driver: mysql
  user: omry
  pass: secret
```

my_app.py

```python
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_name="config")
def my_app(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

if __name__ == "__main__":
    my_app()
```

然后，config.yaml可以自动加载。

```
$ python my_app.py
db:
  driver: mysql
  pass: secret
  user: omry
```

[点击查看更多的应用例子。](https://hydra.cc/docs/intro/)



