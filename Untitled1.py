
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import networkx as nx

G = nx.karate_club_graph()
print("Node Degree")
#for v in G:
 #   print('%s %s' % (v, G.degree(v)))
get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(1)
nx.draw_circular(G, with_labels=True)
plt.show()
plt.cla()
nx.draw_circular(G, with_labels=True)
plt.show() 


# In[2]:


import tensorflow as tf
weights = tf.Variable(tf.random_normal([300, 200], stddev=0.5),
name="weights")


# In[3]:


print(weights)


# In[4]:


x = tf.placeholder(tf.float32, name="x", shape=[None, 784])
W = tf.Variable(tf.random_uniform([784,10], -1, 1), name="W")
multiply = tf.matmul(x, W)
print(multiply)


# In[6]:


b = tf.Variable(tf.zeros([10]), name="biases")
print(b)


# In[8]:


print(tf.zeros([10]))


# In[9]:


print(tf.zeros(10))


# In[11]:


multiply = tf.matmul(x, W)+b
print(multiply)


# In[13]:


kk=list([2,3,4])
print(kk.shape())


# In[14]:


with tf.device('/gpu:2'):
a = tf.constant([1.0, 2.0, 3.0, 4.0], shape=[2, 2], name='a')
b = tf.constant([1.0, 2.0], shape=[2, 1], name='b')
c = tf.matmul(a, b)
sess = tf.Session(config=tf.ConfigProto(
allow_soft_placement=True, log_device_placement=True))
sess.run(c)


# In[17]:


init=tf.constant_initializer(value=0)
b = tf.get_variable("b", [10],initializer=init)
print(b)


# In[1]:


import numpy as np
memory = np.zeros((32, 20 * 2 + 2))
print(memory)


# In[6]:


memory1 = memory[:,2]
print(memory1)


# In[13]:


import tensorflow as tf
import numpy as np

initial_x = np.zeros((4,5))
initial_x[2,4]=1
print(initial_x[:,4].astype(int))


# In[2]:


from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

def plotfunc():
    glClear(GL_COLOR_BUFFER_BIT)
    glutWireTeapot(0.5)
    glutSwapBuffers()

def guimain():
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB)
    glutInitWindowPosition(100, 100)
    glutInitWindowSize(400, 400)
    glutCreateWindow(b"first window")
    glutDisplayFunc(plotfunc)
    glClearColor(0.0, 1.0, 1.0, 0.0)
    gluOrtho2D(-1.0, 1.0, -1.0, 1.0)
    glutMainLoop()

if __name__ == '__main__':
    guimain()


# In[3]:


class FooParent(object):
    def __init__(self):
        self.parent = 'I\'m the parent.'
        print ('Parent')
    
    def bar(self,message):
        print ("%s from Parent" % message)
 
class FooChild(FooParent):
    def __init__(self):
        # super(FooChild,self) 首先找到 FooChild 的父类（就是类 FooParent），然后把类B的对象 FooChild 转换为类 FooParent 的对象
        super(FooChild,self).__init__()    
        print ('Child')
        
    def bar(self,message):
        super(FooChild, self).bar(message)
        print ('Child bar fuction')
        print (self.parent)
 
if __name__ == '__main__':
    fooChild = FooChild()
    fooChild.bar('HelloWorld')


# In[10]:


from numpy import *
a1 =[[1,2,3],[4,5,6]]
a3 = mat(a1)      #列表 ----> 矩阵
print('a3 :',a3)
a4 = a3.tolist()   #矩阵 ---> 列表
print('a4 :',a4)
print('a3 :',a3)
print('a4 :',a4[1])


# In[16]:


b =array([1,2,3,4])
print(b)
b.shape
b.shape=(1,4)
print(b)
b.shape


# In[6]:


import numpy as np
x = np.array([12, 3, 6, 14])
print(x[0])


# In[ ]:


x = np.array([[5, 78, 2, 34, 0],
[6, 79, 3, 35, 1],
[7, 80, 4, 36, 2]])
print(x)
x = np.delete(x, 1,0)
print(x)


# In[ ]:


import numpy
a = numpy.array(([1],[7],[8]))
itemindex = numpy.argwhere(a == [[1],[7]])
print(itemindex)
print(a)
x = numpy.delete(a,itemindex[0,0],0)
print(x)


# In[11]:


from pyflann import *
from numpy import *
from numpy.random import *

dataset = rand(10, 10)
testset = rand(2, 10)
print(dataset)
print(testset)
flann = FLANN()
params = flann.build_index(dataset, algorithm="autotuned", target_precision=0.9, log_level = "info");
print(params)

result, dists = flann.nn_index(testset,5, checks=params["checks"]); 
print(result)
print(dists)


# In[17]:


from pyflann import *
from numpy import *
from numpy.random import *

dataset = array([[0,0],[1,1],[2,2]])
print(*dataset)
print(dataset)
testset =  array([[10,10],[1,1]])
print(testset)
flann = FLANN()
params = flann.build_index(dataset, algorithm="autotuned", target_precision=0.9, log_level = "info");
print(params)

result, dists = flann.nn_index(testset,1, checks=params["checks"]); 
print(result)
print(dists)


# In[9]:


import numpy as np
import random
a = [0,2,3]
a = np.random.randint(1, 5+1)
print(a)
b = np.nonzero(a)
print(b[0])
kk=random.sample(list(b[0]),1)
print(kk)


# In[29]:


from pyflann import *
from numpy import *
from numpy.random import *

dataset = array([[0.0],[1.0],[2.0]])
dataset=dataset.astype(float32)
print(dataset)
testset =  array([4.0])
testset=testset.astype(float32)
print(testset)
flann = FLANN()
params = flann.build_index(dataset, algorithm="autotuned", target_precision=0.9, log_level = "info");
print(params)

result, dists = flann.nn_index(testset,1, checks=params["checks"]); 
print(result)
print(dists)


# In[15]:


range(0, 5.0, 1.0)


# In[24]:


from pyflann import *
from numpy import *
from numpy.random import *

dataset = array([[0],[1],[2]])
print(dataset.astype(float32))


# In[45]:


import networkx as nx
import matplotlib.pyplot as plt
BA= nx.random_graphs.barabasi_albert_graph(20,1)  #生成n=20、m=1的BA无标度网络
plt.cla()
pos = nx.spring_layout(BA)          #定义一个布局，此处采用了spring布局方式
nx.draw(BA,pos,with_labels=False,node_size = 30)  #绘制图形
BA= nx.random_graphs.barabasi_albert_graph(20,1)  #生成n=20、m=1的BA无标度网络


# In[81]:


import networkx as nx
import matplotlib.pyplot as plt

G= nx.Graph()#建立一个空白的图

G.add_node(1)#增加一个叫node1的节点
G.add_node(2)
G.add_node(3)
G.add_node(4)#增加两个叫做1，2的节点
G.add_edge(1,2)#增加一个连接节点1，2的边
G.add_edge(1,3)#增加一个连接节点1，2的边
print(G.nodes())#打印图G的节点
print(G.edges())#打印图G的边
neighbor = G[1].copy()
print(neighbor)

G.remove_edge(1, 2)
print(neighbor)
ba = G

nx.draw(G, hold=True, with_labels=True, node_size=1000)
plt.show()


# In[82]:


n = 20  # 表示节点总数
G = nx.Graph()#建立一个空白的图

nodes = range(n)
G.add_nodes_from(nodes)

edges = [(2, 7), (3, 7),(4, 7), (5, 7), (6, 7), (1, 6), (2, 0)]
G.add_edges_from(edges)
edges = [(8, 7), (8, 9),(8, 10), (10, 11), (10, 12), (10, 13), (10, 14)]
G.add_edges_from(edges)

edges = [(14, 15), (14, 16),(14, 17), (14, 18), (18, 19)]
G.add_edges_from(edges)

# G.remove_edge(1, 3)
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_size=300)  # 绘制图形
plt.show()


# In[14]:


maxgcnodes = list([5,7,8])
allattacks = list([5,7])
Terminal = list(set(maxgcnodes).intersection(set(allattacks)))
if Terminal:
    print(Terminal)
else:
    print(allattacks)


# In[21]:


import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(1)
plt.plot(np.arange(10), np.arange(10), marker="*", linewidth=1, linestyle="--", color="orange")
plt.xlabel('Episodes')
plt.ylabel('Reward per episode')
plt.figure(2)
plt.plot(np.arange(20), 3*np.arange(20), marker="*", linewidth=1, linestyle="--", color="red")
plt.xlabel('Episodes')
plt.ylabel('Reward per episode')
plt.show()  # 显示图表


# In[1]:


import numpy as np
mk = np.ones(4)*(-1e10)
mk[1] = 3
mk = mk[np.newaxis,:]
kkk = mk[0]
print(kkk)
mk[0][1]


# In[14]:


from collections import deque
import numpy as np
a = []
a.append(1)
a.append(2)
print(np.sum(a))


# In[20]:


# coding=utf-8
import tensorflow as tf  

import tensorflow as tf

labels = [[0.2,0.3,0.5],
          [0.1,0.6,0.3]]
logits = [[2,0.5,1],
          [0.1,1,3]]
logits_scaled = tf.nn.softmax(logits)


result1 = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)

result2 = -tf.reduce_sum(labels*tf.log(logits_scaled),1)
result3 = tf.log(logits_scaled)

with tf.Session() as sess:
    print(sess.run(result1))
    print(sess.run(result2))
    print(sess.run(result3))
    print(sess.run(logits_scaled))
    print(sess.run(labels*tf.log(logits_scaled)))


# In[ ]:


import numpy as np

label = np.array([0,3,2,8,9,1])##标签数据，标签从0开始
classes = max(label) + 1
one_hot_label = np.zeros(shape=(label.shape[0],classes))
one_hot_label[np.arange(0,label.shape[0]),label] = 2
print(one_hot_label)
print(np.arange(0,label.shape[0]))


# In[ ]:


kk = list([2,3,4,7])
np.array(kk)
print(np.array(kk))


# In[5]:


import tensorflow as tf

labels = [0,2]

logits = [[2,0.5,1],
          [0.1,1,3]]

logits_scaled = tf.nn.softmax(logits)
logdata = tf.log(logits_scaled)

result1 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)



with tf.Session() as sess:
    print(sess.run(result1))
    print(sess.run(logdata))
    print(sess.run(logits_scaled))


# In[3]:


k='ererere'
m=str(1)
print(k+m)


# In[10]:


di={1: 5, 2: 0, 3: 0, 4: 2, 5: 100}
print(m)
[ v for v in sorted(di.values())]


# In[9]:


m[-4:-1]


# In[7]:


import numpy as np
k = np.zeros((2,4))
k[0,1]=1
k.mean(axis=0)


# In[8]:


import tensorflow as tf
summary_writer = tf.summary.FileWriter('/tmp/test')
summary = tf.Summary(value=[
    tf.Summary.Value(tag="summary_tag", simple_value=0), 
    tf.Summary.Value(tag="summary_tag2", simple_value=1),
])
summary_writer.add_summary(summary, 1)

summary = tf.Summary(value=[
    tf.Summary.Value(tag="summary_tag", simple_value=1), 
    tf.Summary.Value(tag="summary_tag2", simple_value=3),
])
summary_writer.add_summary(summary, 2)

summary_writer.close()


# In[2]:


import numpy as np
k = np.array([3,4,5])
d = np.array([7,3,4])
np.intersect1d(k,d)
np.setdiff1d(k,d)


# In[7]:


d1 = {'a': '7', 'e': '3', 'd': '8', 'g': '7', 'f': '1', 'j': '2', 'l': '9', 'w': '4'}
print(max(zip(d1.values(),d1.keys()))[1])


# In[51]:


import networkx as nx
import numpy as np
import random
alphal = 0.3
kk = np.ones(5)
print(kk)


# In[57]:


import networkx as nx
import numpy as np
import random
a=[]
b=[1,2,3,4]
a.append(1)
a.append(2)
print(b)
print(a)
diffnodes = np.setdiff1d(b, a) 
print(diffnodes)

