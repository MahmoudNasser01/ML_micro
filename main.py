import asyncio

from fastapi import FastAPI, Request

from recognizer import FaceRecognizer
from asgiref.sync import async_to_sync, sync_to_async

app = FastAPI()


async def train(train_data):
    predictor_path = 'shape_predictor_68_face_landmarks.dat'
    recognizer = FaceRecognizer(predictor_path, train_data)
    sync_to_async(recognizer.train('model', 2, 'ball_tree', True))

    return {"status": "done"}


"""
{
       'training_data':
            [
                {
                    'image_path': 'https://media.licdn.com/dms/image/C4D03AQH7OjbTNqDpDw/profile-displayphoto-shrink_400_400/0/1662213868889?e=1692835200&v=beta&t=x2W_6PuRkZNG8GfsdAacvplZ1UagBsCqC7GgfJrw6G0',
                    'label': 'mahmoud'
                },
                {
                    'image_path': 'https://media.licdn.com/dms/image/C5603AQEReSODdvboJw/profile-displayphoto-shrink_400_400/0/1633380703911?e=1692835200&v=beta&t=KSCuROEljCf7jLEi2YOP6Z7TSGEhf0qb_oxi9tdPIUw',
                    'label': 'Alex'
                },
                {
                    'image_path': 'https://media.licdn.com/dms/image/C5603AQEReSODdvboJw/profile-displayphoto-shrink_400_400/0/1633380703911?e=1692835200&v=beta&t=KSCuROEljCf7jLEi2YOP6Z7TSGEhf0qb_oxi9tdPIUw',
                    'label': 'Alex'
                }
            ]
    }
"""


def predict_(paths):
    predictor_path = 'shape_predictor_68_face_landmarks.dat'

    recognizer = FaceRecognizer(predictor_path, {})
    predictions_batch = recognizer.predict_batch(paths, 'model', 0.6)
    for i, image_path in enumerate(paths):
        predictions = predictions_batch[i]
        for name, (top, right, bottom, left) in predictions:
            return name


@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    image_path = data['image_path']
    label = predict_([image_path])
    if label == 'unknown':
        return {"post_id": -1}
    return {"post_id": label}


@app.post("/train")
async def train_(request: Request):
    data = await request.json()
    asyncio.create_task(train(data))  # Create a background task for training
    return {"status": "done"}
