# file: gemini_services.py
import os
import httpx

class GeminiServiceError(Exception):
    pass

async def get_sign_explanation(word: str) -> str:
    """
    Calls the Gemini API to get an explanation for a sign language word.
    Reads API key from GEMINI_API_KEY env var.
    Raises GeminiServiceError on failure for caller to handle.
    """
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise GeminiServiceError("Thiếu GEMINI_API_KEY trong biến môi trường.")

    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={api_key}"

    prompt = (
        f"Provide a simple, step-by-step guide for a beginner on how to perform the sign for '{word}' in American "
        f"Sign Language. Describe the handshape, location, and movement in Vietnamese."
    )

    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    headers = {'Content-Type': 'application/json'}

    timeout = httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            response = await client.post(api_url, json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()

            candidate = result.get("candidates", [{}])
            if not candidate:
                raise GeminiServiceError("Phản hồi AI không có ứng viên.")
            content = candidate[0].get("content", {}).get("parts", [{}])
            if not content:
                raise GeminiServiceError("Phản hồi AI thiếu nội dung.")

            explanation = content[0].get("text")
            if not explanation:
                raise GeminiServiceError("Không thể lấy được giải thích từ AI.")

            return explanation
        except httpx.HTTPStatusError as e:
            raise GeminiServiceError(f"Lỗi API: {e.response.status_code} - {e.response.text}") from e
        except httpx.HTTPError as e:
            raise GeminiServiceError(f"Lỗi mạng khi gọi Gemini: {e}") from e
        except Exception as e:
            raise GeminiServiceError(f"Lỗi không mong muốn: {e}") from e


