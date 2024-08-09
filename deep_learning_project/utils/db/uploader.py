import os

from dotenv import load_dotenv
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload, MediaIoBaseUpload
import io

from utils.log.logger import get_logger

logger = get_logger(__name__)

class GoogleDriveHandler:
    """
    A class for handling Google Drive file upload and download.
    """

    def __init__(self, service):

        load_dotenv()

        self.service_account_file = os.getenv(service)

        self.credentials = service_account.Credentials.from_service_account_file(
            self.service_account_file, scopes=['https://www.googleapis.com/auth/drive.readonly', 'https://www.googleapis.com/auth/drive.file'])

        self.service = build('drive', 'v3', credentials=self.credentials)

    def download_file_from_drive(self, file_id):
        """
        Downloads a file from Google Drive.

        Args:
        file_id (str): The ID of the file to download.

        Returns:
        io.BytesIO: The downloaded file as a BytesIO object.
        """
        request = self.service.files().get_media(fileId=file_id)
        file = io.BytesIO()

        downloader = MediaIoBaseDownload(file, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            logger.info(f'Download {int(status.progress() * 100)}%.')

        logger.info(f'File downloaded successfully to folder ID {file_id}.')


        file.seek(0)
        return file

    def upload_file_to_drive(self, file_path, folder_id):
        """
        Uploads a file to Google Drive.

        Args:
        file_path (str): The path to the file to upload.
        folder_id (str): The ID of the folder to upload the file to.

        Returns:
        None.
        """
        file_metadata = {
            'name': os.path.basename(file_path),
            'parents': [folder_id]
        }
        media = MediaFileUpload(file_path, mimetype='text/csv', resumable=True)
        request = self.service.files().create(body=file_metadata, media_body=media, fields='id')

        response = None
        while response is None:
            status, response = request.next_chunk()
            if status:
                logger.info(f'Upload {int(status.progress() * 100)}% complete.')

        logger.info(f'File {file_path} uploaded successfully to folder ID {folder_id}.')


    def upload_file_from_memory(self, file_bytes, folder_id, filename, mimetype='application/octet-stream'):
        """
        Uploads an in-memory file to Google Drive.

        Args:
        file_bytes (io.BytesIO): The file as a BytesIO object.
        folder_id (str): The ID of the folder to upload the file to.
        filename (str): The name of the file to be saved.
        mimetype (str): The MIME type of the file. Defaults to 'application/octet-stream'.

        Returns:
        None.
        """
        # Upload the in-memory file
        file_metadata = {
            'name': os.path.basename(filename),
            'parents': [folder_id]
        }
        media = MediaIoBaseUpload(file_bytes, mimetype=mimetype, resumable=True)
        request = self.service.files().create(body=file_metadata, media_body=media, fields='id')
        response = None
        while response is None:
            status, response = request.next_chunk()
            if status:
                logger.info(f'Upload {int(status.progress() * 100)}% complete.')

        logger.info(f'File {filename} uploaded successfully to folder ID {folder_id}.')


    def search_file_by_name(self, file_name):
        """
        Searches for a file by name in Google Drive.

        Args:
        file_name (str): The name of the file to search for.

        Returns:
        str: The ID of the file if found, else None.
        """
        results = self.service.files().list(q=f"name='{file_name}'", spaces='drive', fields='files(id, name)').execute()
        items = results.get('files', [])
        if not items:
            logger.info(f'No file found with name: {file_name}')
            return None
        for item in items:
            logger.info(f'Found file: {item["name"]} (ID: {item["id"]})')
            return item['id']


