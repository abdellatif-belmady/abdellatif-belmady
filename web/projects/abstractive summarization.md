---
comments: true
---

## **Introduction**

**``Abstractive Summarization``** is a Natural Language Processing (NLP) task that aims to generate a concise summary of a source text. Unlike **extractive summarization**, Abstractive Summarization doesn't merely copy important sentences from the source text but can also create new, relevant sentences, which can be considered paraphrases. Abstractive Summarization has numerous applications in various domains, from books and literature to science and R&D, financial research, and legal document analysis.

So far, the most recent and effective approach to Abstractive Summarization is to use transformation models specifically tailored to a summary dataset. In this study, we demonstrate how you can easily summarize a text using a powerful model in a few simple steps. First, we'll use two models that are already pre-trained, so no additional training is needed. Then, we'll fine-tune one of these models on our dataset.

Without further ado, let's get started!

## **Importing Data**

```py
import pandas as pd
data = pd.read_json("/content/sample_data/AgrSmall.json")
data.head()
```

## **Using bart-large-cnn & t5-base Transformers**

### **Installing the Transformers Library**

The library we're going to use is Transformers by Huggingface.

To install Transformers, simply run this cell:

```py
pip install transformers
```
!!! Note
    Transformers requires the prior installation of PyTorch. If you haven't already installed PyTorch, visit the official PyTorch website and follow the instructions to install it.

### **Importing Libraries**

After successfully installing Transformers, we can now start importing it into your Python script. We can also import ``os`` to set the environment variable to be used by the GPU in the next step.

```py
from transformers import pipeline
import os
```

Now, we're ready to select the summarization model to use. Huggingface provides two powerful summarization models to use: ``BART`` (bart-large-cnn) and ``t5`` (t5-small, t5-base, t5-large, t5-3b, t5-11b). For more information about these models, please refer to their official documents ([BART document](https://arxiv.org/abs/1910.13461), [t5 document](https://arxiv.org/abs/1910.10683)).

To use the BART model, which is trained on the [CNN/Daily Mail News Dataset](https://www.tensorflow.org/datasets/catalog/cnn_dailymail), we directly use the default parameters via the built-in Huggingface pipeline module:


```py
summarizer = pipeline("summarization")
```

To use the t5 model (e.g., t5-base), trained on the [c4 Common Crawl web corpus](https://www.tensorflow.org/datasets/catalog/c4), we proceed as follows:


```py
summarizer = pipeline("summarization", model="t5-base", tokenizer="t5-base", framework="tf")
```

For more information, please refer to the [Huggingface documentation](https://huggingface.co/transformers/main_classes/pipelines.html#transformers.SummarizationPipeline).

### **Entering Text to Summarize**

Now that our model is ready, we can start selecting the texts we want to summarize. We suggest choosing the first 4 abstracts in our dataset:

We define our variables:

```py
text_1 = data["abstracts"][0]
print(text_1)
```

??? success "Output text_1"
    Most people in rural areas in South Africa (SA) rely on untreated drinking groundwater sources and pit latrine sanitations. A minimum basic sanitation facility should enable safe and appropriate removal of human waste, and although pit latrines provide this, they are still contamination concerns. Pit latrine sludge in SA is mostly emptied and disposed off-site as waste or buried in-situ. Despite having knowledge of potential sludge benefits, most communities in SA are reluctant to use it. This research captured social perceptions regarding latrine sludge management in Monontsha village in the Free State Province of SA through key informant interviews and questionnaires. A key informant interview and questionnaire was done in Monontsha, SA. Eighty participants, representing 5% of all households, were selected. Water samples from four boreholes and four rivers were analyzed for faecal coliforms and E.coli bacteria. On average, five people in a household were sharing a pit latrine. Eighty-three percent disposed filled pit latrines while 17% resorted to closing the filled latrines. Outbreaks of diarrhoea (69%) and cholera (14%) were common. Sixty percent were willing to use treated faecal sludge in agriculture. The binary logistic regression model indicated that predictor variables significantly (p ˂ 0.05) described water quality, faecal sludge management, sludge application in agriculture and biochar adaption. Most drinking water sources in the study had detections ˂ 1 CFU/100 mL. It is therefore imperative to use both qualitative surveys and analytical data. Awareness can go a long way to motivate individuals to adopt to a new change. View Full-Text

```py
text_2 = data["abstracts"][1]
print(text_2)
```

??? success "Output text_2"
    The aim of this study was to highlight the importance of socioeconomic and psychosocial factors in the adoption of sustainable agricultural practices (SAPs) in banana farm production. To this end, data from 300 randomly selected farm households from Pakistan were collected through a structured self-report questionnaire. Using logistic regression (LR) and structural equation modeling (SEM), socioeconomic and psychosocial effects were evaluated. The results show that economic status, watching agricultural training programs, newspaper and radio awareness campaigns, participation in extension programs, perceptions of sustainable agriculture and the feasibility of SAPs were significant factors in farmers’ adoption of sustainable agriculture practices. Also, consistent with the theory of planned behavior (TPB), all its dimensions (attitude, subjective norms and perceived behavioral control) affected the adoption of SAPs. This finding highlights the importance of socioeconomic and psychosocial factors in promoting sustainable agricultural practice among banana production farmers. This is the first study which attempts to provide empirical evidence using a robust procedure (two models—LR and SEM). The practical implication is that, when socioeconomic and psychosocial factors are well supported by satisfactory policy measures, SAP adoption is more than likely, which eventually increases farmers’ adaptive capacity to the changing environment. Ultimately, this leads to sustainable banana production, which has great potential to contribute towards poverty eradication. View Full-Text

```py
text_3 = data["abstracts"][2]
print(text_3)
```

??? success "Output text_3"
    Urban agriculture and gardening provide many health benefits, but the soil is sometimes at risk of heavy metal and metalloid (HMM) contamination. HMM, such as lead and arsenic, can result in adverse health effects for humans. Gardeners may face exposure to these contaminants because of their regular contact with soil and consumption of produce grown in urban areas. However, there is a lack of research regarding whether differential exposure to HMM may be attributed to differential knowledge of exposure sources. In 2018, industrial slag and hazardous levels of soil contamination were detected in West Atlanta. We conducted community-engaged research through surveys and follow-up interviews to understand awareness of slag, HMM in soil, and potential remediation options. Home gardeners were more likely to recognize HMM health effects and to cite health as a significant benefit of gardening than community gardeners. In terms of knowledge, participants were concerned about the potential health effects of contaminants in soil yet unconcerned with produce in their gardens. Gardeners’ knowledge on sources of HMM exposure and methods for remediation were low and varied based on racial group. View Full-Text

```py
text_4 = data["abstracts"][3]
print(text_4)
```

??? success "Output text_4"
    Waste management has become pertinent in urban regions, along with rapid population growth. The current ways of managing waste, such as refuse collection and recycling, are failing to minimise waste in cities. With urban populations growing worldwide, there is the challenge of increased pressure to import food from rural areas. Urban agriculture not only presents an opportunity to explore other means of sustainable food production, but for managing organic waste in cities. However, this opportunity is not taken advantage of. Besides, there is a challenge of mixed reactions from urban planners and policymakers concerning the challenges and benefits presented by using organic waste in urban agriculture. The current paper explores the perceived challenges and opportunities for organic waste utilisation and management through urban agriculture in the Durban South Basin in eThekwini Municipality in KwaZulu-Natal (KZN) Province of South Africa. It is anticipated that this information will be of use to the eThekwini Municipality, policymakers, researchers, urban agriculture initiatives, households and relevant stakeholders in the study areas and similar contexts globally. Two hundred (200) households involved in any urban farming activity and ten (10) key informants (six (6) staff from the Cleaning and Solid Waste Unit of the eThekwini Municipality and four (4) from the urban agricultural initiative) were selected using convenient sampling. Descriptive statistics and inductive thematic analysis were used to analyse data. The significant perceived challenges and risks associated with the utilisation of organic waste through urban agriculture included lack of a supporting policy, climatic variation, lack of land tenure rights, soil contamination and food safety concerns. Qualitative data further showed that the difficulty in segregating waste, water scarcity, difficulty in accessing inputs, limited transportation of organic waste, inadequate handling and treatment of organic waste, and being a health hazard were some important challenges. On the other hand, the significant perceived benefits associated with the utilisation of organic waste through urban agriculture were enhanced food and nutrition security, and opportunities for business incubation. Other important benefits established through qualitative data were an improved market expansion for farmers and improved productivity. Overall, despite the perceived challenges and risks, there is an opportunity to manage organic waste through urban agriculture. It is imperative for an integrated policy encompassing the food, climate and waste management to be developed to support this strategy. All stakeholders—the government, municipal authorities and urban agricultural initiatives should also, guided by the policy, support urban farmers, for example, through pieces of training on how to properly manage and recycle organic waste, land distribution, inputs availability and water usage rights among other things. View Full-Text


### **Summary Generation**

Finally, we can start summarizing the input texts. Here, we specify the minimum and maximum length we want for the summary output and disable sampling to generate fixed summaries. You can do this by running the following commands:

```py
summary_text_1 = summarizer(text, max_length=100, min_length=5, do_sample=False)[0]['summary_text']
print(summary_text_1)
```

There you have it! We get the summary of the first text:

??? success "Output"
    Most people in rural areas in South Africa rely on untreated drinking groundwater sources and pit latrine sanitations . Outbreaks of diarrhoea (69%) and cholera (14%) were common. Sixty percent were willing to use treated faecal sludge in agriculture .

```py
summary_text_2 = summarizer(text_2, max_length=100, min_length=5, do_sample=False)[0]['summary_text']
print(summary_text_2)
```

There you have it! We get the summary of the second text:

??? success "Output"
    The aim of this study was to highlight the importance of socioeconomic and psychosocial factors in the adoption of sustainable agricultural practices (SAPs) in banana farm production . Economic status, watching agricultural training programs, newspaper and radio awareness campaigns, perceptions of sustainable agriculture and the feasibility of SAPs were significant factors .

```py
summary_text_3 = summarizer(text_3, max_length=100, min_length=5, do_sample=False)[0]['summary_text']
print(summary_text_3)
```

There you have it! We get the summary of the third text:

??? success "Output"
    Heavy metal and metalloid (HMM) contamination can result in adverse health effects for humans . In 2018, industrial slag and hazardous levels of soil contamination were detected in West Atlanta . Home gardeners were more likely to recognize HMM health effects than community gardeners .


```py
summary_text_4 = summarizer(text_4, max_length=100, min_length=5, do_sample=False)[0]['summary_text']
print(summary_text_4)
```

There you have it! We get the summary of the third text:

??? success "Output"
    Waste management has become pertinent in urban regions, along with rapid population growth . The current ways of managing waste, such as refuse collection and recycling, are failing to minimise waste in cities . With urban populations growing worldwide, there is the challenge of increased pressure to import food from rural areas .


## **Fine-tuning SimpleT5**

```py
!pip install simplet5
```

```py
import pandas as pd
from sklearn.model_selection import train_test_split

path = "/content/sample_data/AgrSmall.json"
df = pd.read_json(path)
df.head()
```

```py
# simpleT5 expects dataframe to have 2 columns: "source_text" and "target_text"
df = df.rename(columns={"titles":"target_text", "abstracts":"source_text"})
df = df[['source_text', 'target_text']]

# T5 model expects a task related prefix: since it is a summarization task, we will add a prefix "summarize: "
df['source_text'] = "summarize: " + df['source_text']
df
```

```py
train_df, test_df = train_test_split(df, test_size=0.2)
train_df.shape, test_df.shape
```

```py
from simplet5 import SimpleT5

model = SimpleT5()
model.from_pretrained(model_type="t5", model_name="t5-base")
model.train(train_df=train_df[:3000],
            eval_df=test_df[:100], 
            source_max_token_len=128, 
            target_max_token_len=50, 
            batch_size=8, max_epochs=3, use_gpu=True)
```

```py
# let's load the trained model for inferencing:
model.load_model("t5","/content/outputs/simplet5-epoch-0-train-loss-2.806-val-loss-2.5596", use_gpu=True)

text_1 = data["abstracts"][0]
text_2 = data["abstracts"][1]
text_3 = data["abstracts"][2]
text_4 = data["abstracts"][3]
```

```py
model.predict(text_1)
```

??? success "Output"
    ['latrine sludge management in Monontsha, Free State Province of South Africa. Key informant interviews and questionnaires']

```py
model.predict(text_2)
```

??? success "Output"
    ['sustainable agriculture practices among banana production farmers in Pakistan: Evidence from LR and SEM']

```py
model.predict(text_3)
```
??? success "Output"
    ['soil contamination from industrial slag and hazardous levels of metalloid (HMM) contamination in West Atlanta, Georgia. Community-engaged research']

```py
model.predict(text_4)
```
??? success "Output"
    ['challenges and opportunities for organic waste utilisation and management through urban agriculture in the Durban South Basin, KwaZulu-Natal Province of South Africa']
