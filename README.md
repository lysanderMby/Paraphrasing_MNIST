# Paraphrasing_MNIST
*** Work in progress ***

## Quick Start

To train the interpretable classifier and see the intermediate model states (main project output):
'''
python main.py
'''

Implicitly passing inductive biases to machine learning models is currently done with precisely organised model architectures or data pruning methods.

There are sometimes precise models which allow effective, interpretable priors to be used by model to amplify performance. A good example of this is the translation equivariance of (some) convolutional architectures are their impact on image processing. However, it is often difficult or perhaps impossible to directly give inductive priors to models through their architectures.

The workaround is to either alter the training data directly, or add further corrective data in the form of fine tuning once the model has already been trained. Data augmentation such as rotating images in a training set can be used to impart rotational invariance into an image recognition model. Similarly, RLHF is a (considerably more sophisticated) method of providing further training data to correct an existing model.

## Why a Paraphraser?

A paraphrasing model is defined to be a model which converts data to a significantly semantically difference version of itself which is considered to have key properties unchanged. An example is rotating an image, which to a human observer does not change its contents although it does significantly change the individual pixel values passed to a model. 

By creating this transformation, we have the potential to make a model significantly more generalisable. By training on a combination of the original and the paraphrased data, we make a model which shares our approach to irrelevant details. The use of data augmentation in some standard methods have been very well chronicled in papers such as https://arxiv.org/pdf/2204.08610.

So, the use of a paraphraser on training data could be expected to make a model's activations more interpretable by reducing the use of steganography (where uninterpretable thoughts are hidden in plain sight unbeknownst to the human operator). This steganography has the potential to hide unwanted associations from human oversight. For an introduction to this issue in language modelling, see https://www.lesswrong.com/posts/yDcMDJeSck7SuBs24/steganography-in-chain-of-thought-reasoning.

## Designing the Paraphraser

There are two high-level approaches to paraphraser design. The first is a human led RLHF-style design where large scale human input steers an impression of irrelevant details. This 

Note that this work was heavily inspired by the following post - https://www.lesswrong.com/posts/Tzdwetw55JNqFTkzK/why-don-t-we-just-shoggoth-face-paraphraser.
