
import os
import time
from pathlib import Path

import requests


class AssemblyAIProvider:

    MODEL = "universal-3-5-pro"

    def __init__(self):
        self.api_key = os.getenv("ASSEMBLYAI_API_KEY")

        if not self.api_key:
            raise RuntimeError(
                "ASSEMBLYAI_API_KEY is missing from .env"
            )

        region = os.getenv("ASSEMBLYAI_REGION", "us").lower()

        if region == "eu":
            self.base_url = "https://api.eu.assemblyai.com"
        else:
            self.base_url = "https://api.assemblyai.com"

        self.headers = {
            "authorization": self.api_key
        }

    def upload_audio(self, audio_path):

        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(audio_path)

        print("Uploading audio to AssemblyAI...")

        with audio_path.open("rb") as audio_file:

            response = requests.post(
                f"{self.base_url}/v2/upload",
                headers=self.headers,
                data=audio_file,
                timeout=(30, 600)
            )

        if not response.ok:
            raise RuntimeError(
                f"Upload failed: {response.status_code}\n"
                f"{response.text[:2000]}"
            )

        upload_data = response.json()

        if "upload_url" not in upload_data:
            raise RuntimeError(
                "AssemblyAI did not return upload_url."
            )

        print("Audio upload: OK")

        return upload_data["upload_url"]

    def submit_transcription(self, upload_url):

        payload = {
            "audio_url": upload_url,
            "speech_models": [self.MODEL],
            "language_detection": True,
            "speaker_labels": True,
            "speaker_options": {
                "include_speaker_confidence": True
            }
        }

        response = requests.post(
            f"{self.base_url}/v2/transcript",
            headers=self.headers,
            json=payload,
            timeout=60
        )

        if not response.ok:
            raise RuntimeError(
                f"Transcription submission failed: "
                f"{response.status_code}\n"
                f"{response.text[:2000]}"
            )

        result = response.json()

        if "id" not in result:
            raise RuntimeError(
                "AssemblyAI did not return a transcript ID."
            )

        print(f"Transcription job ID: {result['id']}")

        return result["id"]

    def wait_for_result(self, transcript_id):

        endpoint = (
            f"{self.base_url}/v2/transcript/{transcript_id}"
        )

        start_time = time.monotonic()
        timeout_seconds = 3600

        print("Waiting for transcription...")

        while True:

            response = requests.get(
                endpoint,
                headers=self.headers,
                timeout=60
            )

            if not response.ok:
                raise RuntimeError(
                    f"Polling failed: {response.status_code}\n"
                    f"{response.text[:2000]}"
                )

            result = response.json()

            status = result.get("status")

            if status == "completed":

                print("Transcription completed.")

                return result

            if status == "error":

                raise RuntimeError(
                    f"Transcription error: "
                    f"{result.get('error')}"
                )

            if time.monotonic() - start_time > timeout_seconds:

                raise TimeoutError(
                    "Transcription exceeded 30 minutes."
                )

            time.sleep(3)

    def transcribe_file(self, audio_path):

        upload_url = self.upload_audio(audio_path)

        transcript_id = self.submit_transcription(
            upload_url
        )

        result = self.wait_for_result(transcript_id)

        return result
