from fastapi.responses import JSONResponse

def send_response(success: bool, status: int, message: str, data: dict = None):
    """
    Generalized function to structure API responses.
    """
    response = {
        "success": success,
        "status_code": status,
        "message": message,
        "data": data or {}
    }
    return JSONResponse(content=response, status_code=status)

def handle_exception(status: int, message: str, detail: str = None):
    """
    Generalized function to handle exceptions and format error responses.
    """
    response = {
        "success": False,
        "status_code": status,
        "message": message,
        "detail": detail or "An unexpected error occurred."
    }
    print(f"Error: {message} | Detail: {detail}")  

    return JSONResponse(content=response, status_code=status)
