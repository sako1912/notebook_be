import boto3
from botocore.exceptions import ClientError
from fastapi import HTTPException, status
from app.core.config import get_settings
import io

settings = get_settings()

class S3Service:
    def __init__(self):
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
            region_name=settings.aws_region
        )
        self.bucket_name = settings.s3_bucket_name

    async def upload_file(self, file, filename: str) -> str:
        if not file or not file.file:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="유효하지 않은 파일입니다."
            )
            
        try:
            # S3에 파일 업로드
            self.s3_client.upload_fileobj(
                file.file,
                self.bucket_name,
                filename
            )
            
            # 파일의 S3 URL 생성
            url = f"https://{self.bucket_name}.s3.{settings.aws_region}.amazonaws.com/{filename}"
            return url
        
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code == 'NoSuchBucket':
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="S3 버킷을 찾을 수 없습니다."
                )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"S3 업로드 중 오류 발생: {str(e)}"
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="파일 업로드 중 오류가 발생했습니다."
            )

    async def download_file(self, filename: str) -> bytes:
        """
        S3에서 파일을 다운로드합니다.
        """
        try:
            # 파일이 존재하는지 먼저 확인
            try:
                self.s3_client.head_object(Bucket=self.bucket_name, Key=filename)
            except ClientError as e:
                if e.response['Error']['Code'] == '404':
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="해당 경로에 파일을 찾을 수 없습니다."
                    )
                raise e

            # 파일 다운로드
            buffer = io.BytesIO()
            self.s3_client.download_fileobj(
                self.bucket_name,
                filename,
                buffer
            )
            return buffer.getvalue()

        except ClientError as e:
            if not isinstance(e, HTTPException):
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"S3 다운로드 중 오류 발생: {str(e)}"
                )
            raise e
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="파일 다운로드 중 오류가 발생했습니다."
            )

    async def delete_file(self, filename: str) -> bool:
        try:
            # 파일이 존재하는지 먼저 확인
            try:
                self.s3_client.head_object(Bucket=self.bucket_name, Key=filename)
            except ClientError as e:
                if e.response['Error']['Code'] == '404':
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="삭제하려는 파일을 찾을 수 없습니다."
                    )
                raise e

            # S3에서 파일 삭제
            self.s3_client.delete_object(
                Bucket=self.bucket_name,
                Key=filename
            )
            return True
        
        except ClientError as e:
            if not isinstance(e, HTTPException):  # 이미 처리된 404 에러가 아닌 경우
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"S3 삭제 중 오류 발생: {str(e)}"
                )
            raise e
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="파일 삭제 중 오류가 발생했습니다."
            ) 