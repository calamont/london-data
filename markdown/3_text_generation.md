# Generating property listings with AI
<time datetime="2021-04-10 07:00">10th Apr 2021</time>

One of the first thoughts I had after scraping my dataset of rental listings was to train a neural network to generate property descriptions. Ideally, this model could generate these descriptions accurately using information given about the property, such as price, location, and amenities, as opposed to random text. I don't think this automation would be particularly useful for anyone, not even the laziest of estate agents, but it felt like an achievable task given the nature of the data. That was the plan in 2018 and, a mere two years later, I've finally gotten back to it. 

A more formal description of the task I'm attempting is _conditional generation_. That is, getting a machine learning system to generate text _given_ certain input parameters. Having a language model spit out seemingly arbitrary text is easier but far less useful. In contrast, conditional generation could be used for automating the production of quite prescribed quantitative documents, such as quarterly reports.

The recent flurry of deep learning research for NLP has generated impressive gains, with these driven by advances in network architectures (RNN -> LSTM -> Transformers) and colossal training datasets (the Colossal Clean Crawled Corpus, example2**). While the resources required to train state-of-the-art models is beyond the reach of mere mortals, the continued push towards open sourcing such models, and the development of high level APIs to fine-tune them, has meant that even randos writing blogs can join in on the fun. So here we are, but one man with a Macbook Pro, an AWS account, and a datasets of 27,000 rental listings. How far can we get with that?

## Training the model!
To reiterate, the goal was to take the numerical and categorical data I had for the properties in the dataset (e.g. number of bedrooms, rent, amenities), which I will simply refer to as the "property data" from hereon, and use that to train a model to generate text that resembles the original property descriptions as best as possible. I say best as possible, I actually mean as quick as possible, and with minimal inconvenience to me. This means I am using the [transformers library](https://huggingface.co/transformers/) from Huggingface. If you haven't come across this library but are reading this blog I can only assume you are a supportive family member, so I won't go into further details about the library, its [Trainer API](https://huggingface.co/transformers/main_classes/trainer.html), or the [model hub](https://huggingface.co/models). I chose to fine-tune the [T5-large model](https://huggingface.co/transformers/model_doc/t5.html) released by Google. One benefit of this model is that it treats all tasks as a text-to-text problem, where we can define the input text as the property data written as a string, and the output text as the text describing the property. Other common NLP models, such as BERT, are only suited to classification and can't be employed to generate text. I decided to "hashtag-ify" the input data points to give some sort of pattern for the category and associated value. I suspect it would have performed fine without this but hashtags are k00L ok! An example input would looks like...

And the model would try to learn to generate a description that looked like this... (note that this is the actual description, not the AI generated text)

The model was trained using a stripped down script originally from Huggingface, on a p3.2xlarge EC2 instance on AWS, with all default training parameters (see "...with minimal inconvenience to me..."). I was curious to see what we could achieve with such a straight forward implementation.

## Let's see some examples!
So how did the model go? Surprisingly well in fact. I'll cherry pick a few examples here, but feel free to peruse the 80+ test examples I run the model over (that is, examples that the model did not see during the training period).

example1 

example2

example3

In fact, because there is so much general knowledge baked into the model's --- billion parameters from its initial training task, it could actually extrapolate to regions outside of London.

example4

## What's next?
Nothing. I feel that 2 years is long enough to have this on my mind. I consider this case closed. However, feel free to play around with the fine-tuned t5-large model yourself, to make sure I didn't just make these results up. I have chosen not to release the scraped data used in this and the previous two posts. Sometimes users to the websites have listing their full names or phone numbers, and anonymising 27,000 documents was not high on my list of priorities.
