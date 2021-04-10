# Generating reviews with AI
<time datetime="2021-04-10 07:00">10th Apr 2021</time>

One of the first thoughts I had after scraping these rental listings was to train a neural network to generate property descriptions based on some given inputs, such as price, location, amenities. I don't think this would be particularly useful for anyone, not even the laziest of estate agents, but it felt like an achievable task given the nature of the data. I had that thought in 2018 and, a mere two years later, I've finally gotten back to it. 

A more formal description of this task is "conditional generation". That is, getting a machine learning system to generate text given certain input parameters. Having a language model spit out seemingly arbitrary text is easier but far less useful, unless you want to try to sell a collection of the worst short stories written**. In contrast, conditional generation can have many applications, maybe generating quarterly reports or relatively prescribed news articles.

The recent flurry of deep learning research for NLP has generated impressive gains, with these driven by advances in network architectures (RNN -> LSTM -> Transformers) and colosal training datasets (example1, example2**). While the resources required to train state-of-the-art models is beyond the reach of mere mortals, the continued push towards open sourcing such models, and the development of high level APIs to fine-tune them, has meant that even randos writing blogs can join in on the fun. So here we are, a schmuck with a Macbook Pro, an AWS account, and a datasets of 60,000** rental ads. How far can we get with that?

## More details plz...
The goal I set out on was to take the property description of the listings I had, and use that to generate text that would somewhat resemble a description of the property. I can use the thousands of property details and descriptions I have to train a model to best reproduce these descriptions as possible. I say best as possible, but I actually mean as quick as possible, and with as little inconvenience to me. This means I am using the Transformers library from Huggingface. If you haven't come across this library but are reading this blog I can only assume you are a supportive family member, so I won't go into further details about the library, its Trainer API, or the model hub. I chose to fine-tune the T5-large model released by Google. One benefit of this model is that it treats all tasks as a sequence-to-sequence problem, where we can define the input sequence as the property description, and the output sequence as the text describing the property. Other common NLP models, such as BERT, are only suited to classification and can't be employed in this task that requires text generation. I decided to "hashtag-ify" the input properties to give some sort of signal of each of the propreties and their associated value that the model should look out for. I suspect it would have performed fine without this but hashtags are k00L ok! Therefore, an example input would look like this...

And the model would try to learn to generate a description that looked like this...


The model was trained using a stripped down script originally from Huggingface, on a p3.2xlarge EC2 instance on AWS, with all default training parameters (see "as little inconvenience to me"). I was curious to see what we could achieve with such a straight forward implementation...

## Let's see some examples!
So how did the model go? Surprisingly well in fact. I'll cherry pick a few examples here, but feel free to paroose the 80+ test examples I run the model over (that is, examples that the model did not see during the training period). 

example1 

example2

example3

In fact, because there is so much general knowledge baked into the model's --- billion parameters from its initial training task, it could actually extrapolate to regions outside of London.

example4

## What's next?
Nothing. I feel that 2 years is long enough to have this on my mind, festering some might say. I consider this case closed. However, feel free to play around with the fine-tuned t5-large model yourself, to make sure I didn't just make these results up, or even play around with the data (which has been anonymised 
