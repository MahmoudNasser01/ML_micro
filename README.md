# install prdicator file from here
[URL](https://github.com/GuoQuanhao/68_points/blob/master/shape_predictor_68_face_landmarks.dat)

then but it in the root of the project


# create venv

```shell
python3 -m venv venv

```
or
```shell
python -m venv venv
```

# activate venv

```shell
source venv/bin/activate
```

# install requirements
```shell
pip install -r requirements.txt
```


to run the server user

```shell
uvicorn main:app --host 0.0.0.0 --port 5000 --reload
```

# train model
URL``http://0.0.0.0:5000/predict/`` :POST

Body:
```python
{
    "image_path":"https://media.licdn.com/dms/image/C4D03AQH7OjbTNqDpDw/profile-displayphoto-shrink_400_400/0/1662213868889?e=1692835200&v=beta&t=x2W_6PuRkZNG8GfsdAacvplZ1UagBsCqC7GgfJrw6G0"
}
```

# get data
URL:``http://0.0.0.0:5000/train/``

Body:
```python
{
    "training_data": [
        {
            "image_path": "https://media.licdn.com/dms/image/C4D03AQH7OjbTNqDpDw/profile-displayphoto-shrink_400_400/0/1662213868889?e=1692835200&v=beta&t=x2W_6PuRkZNG8GfsdAacvplZ1UagBsCqC7GgfJrw6G0",
            "label": "post_id_1"
        },
        {
            "image_path": "https://media.licdn.com/dms/image/C5603AQEReSODdvboJw/profile-displayphoto-shrink_400_400/0/1633380703911?e=1692835200&v=beta&t=KSCuROEljCf7jLEi2YOP6Z7TSGEhf0qb_oxi9tdPIUw",
            "label": "post_id_2"
        },
        {
            "image_path": "https://media.licdn.com/dms/image/C5603AQEReSODdvboJw/profile-displayphoto-shrink_400_400/0/1633380703911?e=1692835200&v=beta&t=KSCuROEljCf7jLEi2YOP6Z7TSGEhf0qb_oxi9tdPIUw",
            "label": "post_id_3"
        }
    ]
}

```