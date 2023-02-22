## **Introduction**
***Abstractive Summarization*** est une tâche du Natural Language Processing (NLP) qui vise à générer un résumé concis d'un texte source. Contrairement au ***extractive summarization***, Abstractive Summarization ne se contente pas de copier les phrases importantes du texte source, mais peut également en créer de nouvelles qui sont pertinentes, ce qui peut être considéré comme une paraphrase. Abstractive Summarization donne lieu à un certain nombre d'applications dans différents domaines, des livres et de la littérature, à la science et à la R&D, à la recherche financière et à l'analyse de documents juridiques.

Jusqu'à présent, l'approche la plus récente et la plus efficace en matière de Abstractive Summarization consiste à utiliser des modèles de transformation spécifiquement adaptés à un ensemble de données de résumé. Dans cette étude, nous démontrons comment vous pouvez facilement résumer un texte à l'aide d'un modèle puissant en quelques étapes simples. Tout d'abord, nous utiliserons deux modèles qui sont déjà pré-entraînés, de sorte qu'aucun entrainnement supplémentaire n'est nécessaire, puis nous affinerons l'un de ces deux modèles sur notre base de données.

Sans plus attendre, commençons !

## **Importer les données**

```py
import pandas as pd
data = pd.read_json("/content/sample_data/AgrSmall.json")
data.head()
```

## **Utilisation de transformer `bart-large-cnn` & `t5-base`**

### **Installer la bibliothèque Transformers**
La bibliothèque que nous allons utiliser est Transformers par Huggingface.

Pour installer des transformateurs, il suffit d'exécuter cette cellule :

```py
pip install transformers
```
!!! note

    Transformers nécessite l'installation préalable de Pytorch. Si vous n'avez pas encore installé Pytorch, rendez-vous sur [le site officiel de Pytorch](https://pytorch.org/) et suivez les instructions pour l'installer.

### **Importer les bibliothèques**

Après avoir installé transformers avec succès, nous pouvons maintenant commencer à l'importer dans votre script Python. Nous pouvons également importer `os` afin de définir la variable d'environnement à utiliser par le GPU à l'étape suivante.

```py
from transformers import pipeline
import os
```

Maintenant, nous sommes prêts à sélectionner the summarization model à utiliser. Huggingface fournit deux summarization models puissants à utiliser : BART (bart-large-cnn) et t5 (t5-small, t5-base, t5-large, t5-3b, t5-11b). Pour en savoir plus sur ces modèles veuillez consulter leurs documents officiels ([document BART](https://arxiv.org/abs/1910.13461), [document t5](https://arxiv.org/abs/1910.10683)).


Pour utiliser le modèle BART, qui est formé sur le [CNN/Daily Mail News Dataset](https://www.tensorflow.org/datasets/catalog/cnn_dailymail), nous avons utilisés directement les paramètres par défaut via le module intégré Huggingface pipeline :

```py
summarizer = pipeline("summarization")
```

Pour utiliser le modèle t5 (par exemple t5-base), qui est entraîné sur [c4 Common Crawl web corpus](https://www.tensorflow.org/datasets/catalog/c4), nous avons procédé comme suit :

```py
summarizer = pipeline("summarization", model="t5-base", tokenizer="t5-base", framework="tf")
```

Pour plus d'informations, veuillez vous référer à la [Huggingface documentation](https://huggingface.co/transformers/main_classes/pipelines.html#transformers.SummarizationPipeline).

### **Entrer le texte à résumer**

Maintenant que notre modèle est prêt, nous pouvons commencer à choisir les textes que nous voulons résumer. Nous proposons de choisir les 4 premiers abstracts dans notre base de données :

Nous définissons nos variables :

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


### **Génération de résumé**

Enfin, nous pouvons commencer à résumer les textes entrés. Ici, nous déclarons la longueur minimale et la longueur maximale que nous souhaitons pour la sortie des résumés, et nous désactivons également l'échantillonnage pour générer des résumés fixes. Nous pouvons le faire en exécutant les commandes suivantes :

```py
summary_text_1 = summarizer(text, max_length=100, min_length=5, do_sample=False)[0]['summary_text']
print(summary_text_1)
```
Voilà ! Nous obtenons le résumé de premier texte :

??? success "Output"
    Most people in rural areas in South Africa rely on untreated drinking groundwater sources and pit latrine sanitations . Outbreaks of diarrhoea (69%) and cholera (14%) were common. Sixty percent were willing to use treated faecal sludge in agriculture .

```py
summary_text_2 = summarizer(text_2, max_length=100, min_length=5, do_sample=False)[0]['summary_text']
print(summary_text_2)
```
Voilà ! Nous obtenons le résumé de deuxième texte :

??? success "Output"
    The aim of this study was to highlight the importance of socioeconomic and psychosocial factors in the adoption of sustainable agricultural practices (SAPs) in banana farm production . Economic status, watching agricultural training programs, newspaper and radio awareness campaigns, perceptions of sustainable agriculture and the feasibility of SAPs were significant factors .

```py
summary_text_3 = summarizer(text_3, max_length=100, min_length=5, do_sample=False)[0]['summary_text']
print(summary_text_3)
```

Voilà ! Nous obtenons le résumé de troisième texte :

??? success "Output"
    Heavy metal and metalloid (HMM) contamination can result in adverse health effects for humans . In 2018, industrial slag and hazardous levels of soil contamination were detected in West Atlanta . Home gardeners were more likely to recognize HMM health effects than community gardeners .


```py
summary_text_4 = summarizer(text_4, max_length=100, min_length=5, do_sample=False)[0]['summary_text']
print(summary_text_4)
```
Voilà ! Nous obtenons le résumé de quatrième texte :

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