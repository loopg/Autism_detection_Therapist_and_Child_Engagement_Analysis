def predict_gaze(frame_tensor):
    # Load pre-trained CPU-friendly gaze estimation model (e.g., Detectron2 with appropriate weights)
    model = model_zoo.get("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    # model = model_zoo.get("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml", base="detectron2://coco2017-instances")  # For remote model loading
    model.eval()

    # Convert frame_tensor to NumPy array and preprocess (if needed)
    image = frame_tensor.numpy()
    image = preprocess_image_for_gaze_estimation(image)  # Adapt for gaze model

    with torch.no_grad():  # Disable gradient calculation for faster inference
        predictions = model([image])[0]

        # Extract gaze prediction from the model output (replace with specific logic)
        gaze_boxes = predictions["instances"].pred_boxes.tensor.cpu().numpy()  # Assuming bounding boxes
        gaze_directions = extract_gaze_from_boxes(gaze_boxes)  # Placeholder for gaze extraction

        # Modify for separate child and therapist gaze (replace with logic)
        child_gaze, therapist_gaze = separate_gaze_predictions(gaze_directions)

        return child_gaze, therapist_gaze  # Or modify to return combined gaze info

