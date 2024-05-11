from mlserver.types import InferenceRequest, InferenceResponse, ResponseOutput
from ultralytics import YOLO
import base64
from PIL import Image
import numpy as np
from io import BytesIO
import json
import mlserver

class YOLOv8Model(mlserver.MLModel):
    async def load(self) -> bool:
        self._model = YOLO('yolov8n.pt')  # load an official model
        mlserver.register("count_obj_detection", "This is a count objects detection")
        return await super().load()

    async def predict(self, payload: InferenceRequest) -> InferenceResponse:
        img = self.base64_to_img(payload.inputs[0].data[0])
        results = self._model(img)
        response_outputs = []
        
        for result in results:
            # Each output part could be a different ResponseOutput object
            response_outputs.append(ResponseOutput(
                name="boxes",
                shape=result.boxes.xywh.shape,
                datatype="FP32",
                data=self.serialize_numpy(result.boxes.xywh)
            ))
            response_outputs.append(ResponseOutput(
                name="probs",
                shape=result.boxes.conf.shape,
                datatype="FP32",
                data=self.serialize_numpy(result.boxes.conf)
            ))
            response_outputs.append(ResponseOutput(
                name="cls",
                shape=result.boxes.cls.shape,
                datatype="FP32",
                data=self.serialize_numpy(result.boxes.cls)
            ))
        mlserver.log(count_obj_detection=len(response_outputs))
        return InferenceResponse(
                model_name="yolov8n",
                model_version="v1",
                outputs=response_outputs
            )

    
    def base64_to_img(self, b64str):
        """Convert base64 string to PIL image."""
        image_data = base64.b64decode(b64str)
        image = Image.open(BytesIO(image_data))
        image = image.convert('RGB')  # Ensure image is in RGB format
        return np.array(image)  # Return image as a numpy array

    def serialize_numpy(self, np_array):
        return json.dumps(np.array(np_array).tolist())
