# EnrichEvent
Official implementation of "[**EnrichEvent: Enriching Social Data with Contextual Information for Emerging Event Extraction**](https://arxiv.org/abs/2307.16082)"

## Introduction
Social platforms have emerged as crucial platforms for disseminating information and discussing real-life social events, offering researchers an excellent opportunity to design and implement novel event detection frameworks. However, most existing approaches only exploit keyword burstiness or network structures to detect unspecified events. Thus, they often need help identifying unknown events regarding the challenging nature of events and social data. Social data, e.g., tweets, is characterized by misspellings, incompleteness, word sense ambiguation, irregular language, and variation in aspects of opinions. Moreover, extracting discriminative features and patterns for evolving events by exploiting the limited structural knowledge is almost infeasible. To address these challenges, in this paper, we propose a novel framework, namely EnrichEvent, that leverages the linguistic and contextual representations of streaming social data. In particular, we leverage contextual and linguistic knowledge to detect semantically related tweets and enhance the effectiveness of the event detection approaches. Eventually, our proposed framework produces cluster chains for each event to show the evolving variation of the event through time. We conducted extensive experiments to evaluate our framework, validating its high performance and effectiveness in detecting and distinguishing unspecified social events.

## Inputs & Outputs
- **Input**: Streams of message blocks.
- **Output**: Existing events presented as cluster chains.

## How to Run
1. Open `main.ipynb`.
2. Initialize and customize the parameters based on your requirements.
3. Run all cells in `main.ipynb`.
4. The results will be saved in the specified output directory.

## About Dataset
1. You can find the details of our proposed datasets in the `/Dataset` folder.
   - **Note**: You may also use your own dataset, but ensure its structure and column names are compatible with the model.

## Training the Trend Detection Model
1. Navigate to the `/Trend_Detection` folder.
2. Use `train.py` to build and train the trend detection model.
   - **Note**: A labeled dataset is required. You can use `dataset_labeling.py` to label your dataset based on key phrases.

## Training the Event Summarization Model
1. Navigate to the `/Event_Summarization` folder.
2. Use `train.py` to build and train the event summarization model.
   - **Note**: A pre-trained embedding model is required based on the language of your dataset.

## Citation
For more details, please refer to our paper:

```
@article{Esfahani2023EnrichEvent,
  title={EnrichEvent: Enriching Social Data with Contextual Information for Emerging Event Extraction},
  author={Mohammadali Sefidi Esfahani and Mohammad Akbari},
  journal={Arxiv},
  year={2023},
  doi={https://arxiv.org/abs/2307.16082}
}
```

---

For any questions or issues, feel free to contact us.

---
