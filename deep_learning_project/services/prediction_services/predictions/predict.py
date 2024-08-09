from services.prediction_services.predictions.setup.test_predictor import TestPredictor

if __name__ == '__main__':
    """
    Main script for making predictions using the AmountPredictor and TestPredictor.

    This script loads the data, initializes the predictor, makes predictions for each unique shop_id and item_id combination,
    and saves the predictions to a text file.
    """


    test_predictor = TestPredictor(data_path='LAST_FILE_NAME',
                                   test_path='TEST_FILE_ID',
                                   service='SERVICE_ACCOUNT_FILE',
                                   model_file_name='MODEL_NAME',
                                   scaler_file_name='SCALER_NAME',
                                   pca_file_name='PCA_NAME',
                                   output_id='UPLOAD_DRIVE_FOLDER',
                                   output_path='PREDICTED_AMOUNT')
    test_predictor.run()

    test_predictor.make_api_prediction(101, 107172)