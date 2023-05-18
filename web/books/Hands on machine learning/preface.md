# **Preface**

## **The Machine Learning Tsunami**

In 2006, Geoffrey Hinton et al. published a paper [^1] showing how to train a deep neural network capable of recognizing handwritten digits with state-of-the-art precision (>98%). They branded this technique “Deep Learning.” Training a deep neural net was widely considered impossible at the time,[^2] and most researchers had abandoned the idea since the 1990s. This paper revived the interest of the scientific community and before long many new papers demonstrated that Deep Learning was not only possible, but capable of mind-blowing achievements that no other Machine Learning (ML) technique could hope to match (with the help of tremendous computing power and great amounts of data). This enthusiasm soon extended to many other areas of Machine Learning.

Fast-forward 10 years and Machine Learning has conquered the industry: it is now at the heart of much of the magic in today’s high-tech products, ranking your web search results, powering your smartphone’s speech recognition, recommending videos, and beating the world champion at the game of Go. Before you know it, it will be driving your car.

## **Machine Learning in Your Projects**

So naturally you are excited about Machine Learning and you would love to join the party!

Perhaps you would like to give your homemade robot a brain of its own? Make it recognize faces? Or learn to walk around?

Or maybe your company has tons of data (user logs, financial data, production data, machine sensor data, hotline stats, HR reports, etc.), and more than likely you could unearth some hidden gems if you just knew where to look; for example:

- Segment customers and find the best marketing strategy for each group

- Recommend products for each client based on what similar clients bought

-  Detect which transactions are likely to be fraudulent

- Forecast next year’s revenue

- [And more](https://www.kaggle.com/wiki/DataScienceUseCases)

Whatever the reason, you have decided to learn Machine Learning and implement it in your projects. Great idea!

## **Objective and Approach**

This book assumes that you know close to nothing about Machine Learning. Its goal is to give you the concepts, the intuitions, and the tools you need to actually implement programs capable of <em>learning from data.</em>

We will cover a large number of techniques, from the simplest and most commonly used (such as linear regression) to some of the Deep Learning techniques that regularly win competitions.

Rather than implementing our own toy versions of each algorithm, we will be using actual production-ready Python frameworks:

- [Scikit-Learn](http://scikit-learn.org/) is very easy to use, yet it implements many Machine Learning algorithms efficiently, so it makes for a great entry point to learn Machine Learning.

- [TensorFlow](https://tensorflow.org/) is a more complex library for distributed numerical computation. It makes it possible to train and run very large neural networks efficiently by distributing the computations across potentially hundreds of multi-GPU servers. TensorFlow was created at Google and supports many of their large-scale Machine Learning applications. It was open sourced in November 2015.

- [Keras](https://keras.io/) is a high level Deep Learning API that makes it very simple to train and run neural networks. It can run on top of either TensorFlow, Theano or Microsoft Cognitive Toolkit (formerly known as CNTK). TensorFlow comes with its own implementation of this API, called <em>tf.keras</em>, which provides support for some advanced TensorFlow features (e.g., to efficiently load data).

The book favors a hands-on approach, growing an intuitive understanding of Machine Learning through concrete working examples and just a little bit of theory. While you can read this book without picking up your laptop, we highly recommend you experiment with the code examples available online as Jupyter notebooks at [https://github.com/ageron/handson-ml2](https://github.com/ageron/handson-ml2).


## **Prerequisites**

This book assumes that you have some Python programming experience and that you are familiar with Python’s main scientific libraries, in particular [NumPy](http://numpy.org/), [Pandas](http://pandas.pydata.org/), and [Matplotlib](http://matplotlib.org/).

Also, if you care about what’s under the hood you should have a reasonable understanding of college-level math as well (calculus, linear algebra, probabilities, and statistics).

If you don’t know Python yet, [http://learnpython.org/](http://learnpython.org/) is a great place to start. The official tutorial on [python.org](https://docs.python.org/3/tutorial/) is also quite good.

If you have never used Jupyter, ***Chapter 2*** will guide you through installation and the basics: it is a great tool to have in your toolbox.

If you are not familiar with Python’s scientific libraries, the provided Jupyter notebooks include a few tutorials. There is also a quick math tutorial for linear algebra.

## **Roadmap**

This book is organized in two parts. ***The Fundamentals of Machine Learning***, covers the following topics:

-  What is Machine Learning? What problems does it try to solve? What are the main categories and fundamental concepts of Machine Learning systems?

- The main steps in a typical Machine Learning project.

-  Learning by fitting a model to data.

- Optimizing a cost function.

- Handling, cleaning, and preparing data.

- Selecting and engineering features.

- The main challenges of Machine Learning, in particular underfitting and overfitting (the bias/variance tradeoff).

- Reducing the dimensionality of the training data to fight the curse of dimensionality.

- Other unsupervised learning techniques, including clustering, density estimation
and anomaly detection.

- The most common learning algorithms: Linear and Polynomial Regression, Logistic Regression, k-Nearest Neighbors, Support Vector Machines, Decision Trees, Random Forests, and Ensemble methods.

***Neural Networks and Deep Learning***, covers the following topics:

- What are neural nets? What are they good for?

- The most important neural net architectures: feedforward neural nets, convolutional nets, recurrent nets, long short-term memory (LSTM) nets, autoencoders and generative adversarial networks (GANs).

- Techniques for training deep neural nets.

- Scaling neural networks for large datasets.

- Learning strategies with Reinforcement Learning.

- Handling uncertainty with Bayesian Deep Learning.

The first part is based mostly on Scikit-Learn while the second part uses TensorFlow and Keras.

!!! warning annotate "Warning"

    Don’t jump into deep waters too hastily: while Deep Learning is no doubt one of the most exciting areas in Machine Learning, you should master the fundamentals first. Moreover, most problems can be solved quite well using simpler techniques such as Random Forests and Ensemble methods (discussed in ***The Fundamentals of Machine Learning***). Deep Learning is best suited for complex problems such as image recognition, speech recognition, or natural language processing, provided you have enough data, computing power, and patience.


## **Other Resources**

Many resources are available to learn about Machine Learning. Andrew Ng’s [ML course on Coursera](https://www.coursera.org/learn/machine-learning/) and Geoffrey Hinton’s [course on neural networks and Deep Learning](https://www.coursera.org/course/neuralnets) are amazing, although they both require a significant time investment (think months).

There are also many interesting websites about Machine Learning, including of course Scikit-Learn’s exceptional [User Guide](http://scikit-learn.org/stable/user_guide.html). You may also enjoy [Dataquest](https://www.dataquest.io/), which provides very nice interactive tutorials, and ML blogs such as those listed on [Quora](https://homl.info/1). Finally, the [Deep Learning website](http://deeplearning.net/) has a good list of resources to learn more.

Of course there are also many other introductory books about Machine Learning, in
particular:

- Joel Grus, [Data Science from Scratch](http://shop.oreilly.com/product/0636920033400.do) (O’Reilly). This book presents the fundamentals of Machine Learning, and implements some of the main algorithms in pure Python (from scratch, as the name suggests).

- Stephen Marsland, <em>Machine Learning: An Algorithmic Perspective</em> (Chapman and Hall). This book is a great introduction to Machine Learning, covering a wide range of topics in depth, with code examples in Python (also from scratch, but using NumPy).

- Sebastian Raschka, <em>Python Machine Learning</em> (Packt Publishing). Also a great
introduction to Machine Learning, this book leverages Python open source libra‐
ries (Pylearn 2 and Theano).

- François Chollet, <em>Deep Learning with Python</em> (Manning). A very practical book
that covers a large range of topics in a clear and concise way, as you might expect
from the author of the excellent Keras library. It favors code examples over math‐
ematical theory.

- Yaser S. Abu-Mostafa, Malik Magdon-Ismail, and Hsuan-Tien Lin, <em>Learning from Data</em> (AMLBook). A rather theoretical approach to ML, this book provides deep insights, in particular on the bias/variance tradeoff (see ***Training Models***).

- Stuart Russell and Peter Norvig, <em>Artificial Intelligence: A Modern Approach, 3rd
Edition</em> (Pearson). This is a great (and huge) book covering an incredible amount
of topics, including Machine Learning. It helps put ML into perspective.

Finally, a great way to learn is to join ML competition websites such as [Kaggle.com](https://www.kaggle.com/) this will allow you to practice your skills on real-world problems, with help and insights from some of the best ML professionals out there.

## **Conventions Used in This Book**

The following typographical conventions are used in this book:
<em>Italic</em>
    Indicates new terms, URLs, email addresses, filenames, and file extensions.
Constant width
    Used for program listings, as well as within paragraphs to refer to program elements such as variable or function names, databases, data types, environment variables, statements and keywords.
<bold>Constant width bold</bold>
    Shows commands or other text that should be typed literally by the user.
<em>Constant width italic</em>
    Shows text that should be replaced with user-supplied values or by values determined by context.

!!! tip "Tip"

    This element signifies a tip or suggestion.

!!! note "Note"

    This element signifies a general note.

!!! warning "Warning"

    This element indicates a warning or caution.

## **Code Examples**

Supplemental material (code examples, exercises, etc.) is available for download at [https://github.com/ageron/handson-ml2](https://github.com/ageron/handson-ml2). It is mostly composed of Jupyter notebooks.

Some of the code examples in the book leave out some repetitive sections, or details that are obvious or unrelated to Machine Learning. This keeps the focus on the important parts of the code, and it saves space to cover more topics. However, if you want the full code examples, they are all available in the Jupyter notebooks.

Note that when the code examples display some outputs, then these code examples are shown with Python prompts (>>> and ...), as in a Python shell, to clearly distinguish the code from the outputs. For example, this code defines the square() function then it computes and displays the square of 3:

``` py
def square(x):
    return x ** 2
...
result = square(3)
result
```
9

When code does not display anything, prompts are not used. However, the result may
sometimes be shown as a comment like this:

``` py
def square(x):
    return x ** 2
    
result = square(3)
result # result is 9
```

## **Using Code Examples**

This book is here to help you get your job done. In general, if example code is offered with this book, you may use it in your programs and documentation. You do not need to contact us for permission unless you’re reproducing a significant portion of the code. For example, writing a program that uses several chunks of code from this book does not require permission. Selling or distributing a CD-ROM of examples from O’Reilly books does require permission. Answering a question by citing this book and quoting example code does not require permission. Incorporating a significant amount of example code from this book into your product’s documentation does require permission.

We appreciate, but do not require, attribution. An attribution usually includes the title, author, publisher, and ISBN. For example: “Hands-On Machine Learning with Scikit-Learn, Keras and TensorFlow by Aurélien Géron (O’Reilly). Copyright 2019 Aurélien Géron, 978-1-492-03264-9.” If you feel your use of code examples falls outside fair use or the permission given above, feel free to contact us at [permissions@oreilly.com](permissions@oreilly.com).

## **O’Reilly Safari**

[Safari](http://oreilly.com/safari) (formerly Safari Books Online) is a membership-based training and reference platform for enterprise, government, educators, and individuals.

Members have access to thousands of books, training videos, Learning Paths, interactive tutorials, and curated playlists from over 250 publishers, including O’Reilly Media, Harvard Business Review, Prentice Hall Professional, Addison-Wesley Professional, Microsoft Press, Sams, Que, Peachpit Press, Adobe, Focal Press, Cisco Press, John Wiley & Sons, Syngress, Morgan Kaufmann, IBM Redbooks, Packt, Adobe
Press, FT Press, Apress, Manning, New Riders, McGraw-Hill, Jones & Bartlett, and Course Technology, among others.

For more information, please visit [http://oreilly.com/safari](http://oreilly.com/safari).

## **How to Contact Us**

Please address comments and questions concerning this book to the publisher:

:   O’Reilly Media, Inc.

    1005 Gravenstein Highway North

    Sebastopol, CA 95472

    800-998-9938 (in the United States or Canada)

    707-829-0515 (international or local)

    707-829-0104 (fax)

We have a web page for this book, where we list errata, examples, and any additional information. You can access this page at [http://bit.ly/hands-on-machine-learning-with-scikit-learn-and-tensorflow](http://bit.ly/hands-on-machine-learning-with-scikit-learn-and-tensorflow) or [https://homl.info/oreilly](https://homl.info/oreilly).

To comment or ask technical questions about this book, send email to [bookques‐tions@oreilly.com](bookques‐tions@oreilly.com).

For more information about our books, courses, conferences, and news, see our website at [http://www.oreilly.com](http://www.oreilly.com).

Find us on Facebook: [http://facebook.com/oreilly](http://facebook.com/oreilly)

Follow us on Twitter: [http://twitter.com/oreillymedia](http://twitter.com/oreillymedia)

Watch us on YouTube: [http://www.youtube.com/oreillymedia](http://www.youtube.com/oreillymedia)

## **Changes in the Second Edition**

This second edition has five main objectives:

1. Cover additional topics: additional unsupervised learning techniques (including clustering, anomaly detection, density estimation and mixture models), additional techniques for training deep nets (including self-normalized networks), additional computer vision techniques (including the Xception, SENet, object
detection with YOLO, and semantic segmentation using R-CNN), handling sequences using CNNs (including WaveNet), natural language processing using RNNs, CNNs and Transformers, generative adversarial networks, deploying TensorFlow models, and more.

2. Update the book to mention some of the latest results from Deep Learning research.

3. Migrate all TensorFlow chapters to TensorFlow 2, and use TensorFlow’s implementation of the Keras API (called tf.keras) whenever possible, to simplify the code examples.

4. Update the code examples to use the latest version of Scikit-Learn, NumPy, Pandas, Matplotlib and other libraries.

5. Clarify some sections and fix some errors, thanks to plenty of great feedback from readers.

Some chapters were added, others were rewritten and a few were reordered. ***Table P-1*** shows the mapping between the 1st edition chapters and the 2nd edition chapters:

<em>***Table P-1.*** Chapter mapping between 1st and 2nd edition</em>

| 1st Ed.chapter    | 2nd Ed.chapter      | % Changes      | 2nd Ed.chapter                     
| ----------------- | ------------------- |----------------|----------------------- 
| 1                 | 1                   | <10%           | The Machine Learning Landscape
| 2                 | 2                   | <10%           | End-to-End Machine Learning Project
| 3                 | 3                   | <10%           | Classification
| 4                 | 4                   | <10%           | Training Models
| 5                 | 5                   | <10%           | Support Vector Machines
| 6                 | 6                   | <10%           | Decision Trees
| 7                 | 7                   | <10%           | Ensemble Learning and Random Forests
| 8                 | 8                   | <10%           | Dimensionality Reduction
| N/A               | 9                   | 100% new       | Unsupervised Learning Techniques
| 10                | 10                  | ~75%           | Introduction to Artificial Neural Networks with Keras
| 11                | 11                  | ~50%           | Training Deep Neural Networks
| 9                 | 12                  | 100% rewritten | Custom Models and Training with TensorFlow
| Part of 12        | 13                  | 100% rewritten | Loading and Preprocessing Data with TensorFlow
| 13                | 14                  | ~50%           | Deep Computer Vision Using Convolutional Neural Networks
| Part of 14        | 15                  | ~75%           | Processing Sequences Using RNNs and CNNs
| Part of 14        | 16                  | ~90%           | Natural Language Processing with RNNs and Attention
| 15                | 17                  | ~75%           | Autoencoders and GANs
| 16                | 18                  | ~75%           | Reinforcement Learning
| Part of 12        | 19                  | 100% rewritten | Deploying your TensorFlow Models

More specifically, here are the main changes for each 2nd edition chapter (other than clarifications, corrections and code updates):

- Chapter 1
     * Added a section on handling mismatch between the training set and the validation & test sets.
    
- Chapter 2
    *  Added how to compute a confidence interval.
    *  Improved the installation instructions (e.g., for Windows).
    *  Introduced the upgraded OneHotEncoder and the new ColumnTransformer.

- Chapter 4
    *  Explained the need for training instances to be Independent and Identically Distributed (IID).

- Chapter 7
    *  Added a short section about XGBoost.

- Chapter 9 – new chapter including:
    *  Clustering with K-Means, how to choose the number of clusters, how to use it for dimensionality reduction, semi-supervised learning, image segmentation, and more.
    *  Gaussian mixture models, the Expectation-Maximization (EM) algorithm, Bayesian variational inference, and how mixture models can be used for clustering, density estimation, anomaly detection and novelty detection.
    *  Overview of other anomaly detection and novelty detection algorithms.

- Chapter 10 (mostly new)
    *  Added an introduction to the Keras API, including all its APIs (Sequential, Functional and Subclassing), persistence and callbacks (including the Tensor Board callback).

- Chapter 11 (many changes)
    *  Introduced self-normalizing nets, the SELU activation function and Alpha Dropout.
    *  Introduced self-supervised learning.
    *  Added Nadam optimization.
    *  Added Monte-Carlo Dropout.
    *  Added a note about the risks of adaptive optimization methods.
    *  Updated the practical guidelines.

- Chapter 12 – completely rewritten chapter, including:
    *  A tour of TensorFlow 2
    *  TensorFlow’s lower-level Python API
    *  Writing custom loss functions, metrics, layers, models
    *  Using auto-differentiation and creating custom training algorithms.
    *  TensorFlow Functions and graphs (including tracing and autograph).

- Chapter 13 – new chapter, including:
    *  The Data API
    *  Loading/Storing data efficiently using TFRecords
    *  The Features API (including an introduction to embeddings).
    *  An overview of TF Transform and TF Datasets
    *  Moved the low-level implementation of the neural network to the exercises.
    *  Removed details about queues and readers that are now superseded by the Data API.

- Chapter 14
    *  Added Xception and SENet architectures.
    *  Added a Keras implementation of ResNet-34.
    *  Showed how to use pretrained models using Keras.
    *  Added an end-to-end transfer learning example.
    *  Added classification and localization.
    *  Introduced Fully Convolutional Networks (FCNs).
    *  Introduced object detection using the YOLO architecture.
    *  Introduced semantic segmentation using R-CNN.

- Chapter 15
    *  Added an introduction to Wavenet.
    *  Moved the Encoder–Decoder architecture and Bidirectional RNNs to Chapter 16.

- Chapter 16
    *  Explained how to use the Data API to handle sequential data.
    *  Showed an end-to-end example of text generation using a Character RNN, using both a stateless and a stateful RNN.
    *  Showed an end-to-end example of sentiment analysis using an LSTM.
    *  Explained masking in Keras.
    *  Showed how to reuse pretrained embeddings using TF Hub.
    *  Showed how to build an Encoder–Decoder for Neural Machine Translation using TensorFlow Addons/seq2seq.
    *  Introduced beam search.
    *  Explained attention mechanisms.
    *  Added a short overview of visual attention and a note on explainability.
    *  Introduced the fully attention-based Transformer architecture, including positional embeddings and multi-head attention.
    *  Added an overview of recent language models (2018).

- Chapters 17, 18 and 19: coming soon.

## **Acknowledgments**

Never in my wildest dreams did I imagine that the first edition of this book would get such a large audience. I received so many messages from readers, many asking questions, some kindly pointing out errata, and most sending me encouraging words. I cannot express how grateful I am to all these readers for their tremendous support. Thank you all so very much! Please do not hesitate to [file issues on github](https://github.com/ageron/handson-ml2/issues) if you find errors in the code examples (or just to ask questions), or to submit [errata](https://homl.info/errata2) if you find errors in the text. Some readers also shared how this book helped them get their first job, or how it helped them solve a concrete problem they were working on: I find such feedback incredibly motivating. If you find this book helpful, I would love it if you could share your story with me, either privately (e.g., via [LinkedIn](https://www.linkedin.com/in/aurelien-geron/)) or publicly (e.g., in an [Amazon review](https://homl.info/amazon2)).

I am also incredibly thankful to all the amazing people who took time out of their busy lives to review my book with such care. In particular, I would like to thank François Chollet for reviewing all the chapters based on Keras & TensorFlow, and giving me some great, in-depth feedback. Since Keras is one of the main additions to this 2nd edition, having its author review the book was invaluable. I highly recommend François’s excellent book [Deep Learning with Python](https://homl.info/cholletbook)[#3]: it has the conciseness, clarity and depth of the Keras library itself. Big thanks as well to Ankur Patel, who reviewed every chapter of this 2nd edition and gave me excellent feedback.

This book also benefited from plenty of help from members of the TensorFlow team, in particular Martin Wicke, who tirelessly answered dozens of my questions and dispatched the rest to the right people, including Alexandre Passos, Allen Lavoie, André Susano Pinto, Anna Revinskaya, Anthony Platanios, Clemens Mewald, Dan Moldo‐van, Daniel Dobson, Dustin Tran, Edd Wilder-James, Goldie Gadde, Jiri Simsa, Karmel Allison, Nick Felt, Paige Bailey, Pete Warden (who also reviewed the 1st edition), Ryan Sepassi, Sandeep Gupta, Sean Morgan, Todd Wang, Tom O’Malley, William Chargin, and Yuefeng Zhou, all of whom were tremendously helpful. A huge thank you to all of you, and to all other members of the TensorFlow team. Not just for your help, but also for making such a great library.

Big thanks to Haesun Park, who gave me plenty of excellent feedback and caught several errors while he was writing the Korean translation of the 1st edition of this book. He also translated the Jupyter notebooks to Korean, not to mention TensorFlow’s documentation. I do not speak Korean, but judging by the quality of his feedback, all his translations must be truly excellent! Moreover, he kindly contributed some of the solutions to the exercises in this book.

Many thanks as well to O’Reilly’s fantastic staff, in particular Nicole Tache, who gave me insightful feedback, always cheerful, encouraging, and helpful: I could not dream of a better editor. Big thanks to Michele Cronin as well, who was very helpful (and patient) at the start of this 2nd edition. Thanks to Marie Beaugureau, Ben Lorica, Mike Loukides, and Laurel Ruma for believing in this project and helping me define its scope. Thanks to Matt Hacker and all of the Atlas team for answering all my technical
questions regarding formatting, asciidoc, and LaTeX, and thanks to Rachel Monaghan, Nick Adams, and all of the production team for their final review and their hundreds of corrections.

I would also like to thank my former Google colleagues, in particular the YouTube video classification team, for teaching me so much about Machine Learning. I could never have started the first edition without them. Special thanks to my personal ML gurus: Clément Courbet, Julien Dubois, Mathias Kende, Daniel Kitachewsky, James Pack, Alexander Pak, Anosh Raj, Vitor Sessak, Wiktor Tomczak, Ingrid von Glehn,
Rich Washington, and everyone I worked with at YouTube and in the amazing Google research teams in Mountain View. All these people are just as nice and helpful as they are bright, and that’s saying a lot.

I will never forget the kind people who reviewed the 1st edition of this book, including David Andrzejewski, Eddy Hung, Grégoire Mesnil, Iain Smears, Ingrid von Glehn, Justin Francis, Karim Matrah, Lukas Biewald, Michel Tessier, Salim Sémaoune, Vincent Guilbeau and of course my dear brother Sylvain.

Last but not least, I am infinitely grateful to my beloved wife, Emmanuelle, and to our three wonderful children, Alexandre, Rémi, and Gabrielle, for encouraging me to work hard on this book, as well as for their insatiable curiosity: explaining some of the most difficult concepts in this book to my wife and children helped me clarify my thoughts and directly improved many parts of this book. Plus, they keep bringing me cookies and coffee! What more can one dream of?


[^1]:
Available on Hinton’s home page at [http://www.cs.toronto.edu/~hinton/](http://www.cs.toronto.edu/~hinton/).

[^2]:
Despite the fact that Yann Lecun’s deep convolutional neural networks had worked well for image recognition since the 1990s, although they were not as general purpose.

[#3]:
    “Deep Learning with Python,” François Chollet (2017).
















