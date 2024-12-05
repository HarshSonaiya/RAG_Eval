def send_response(success: bool, status: int, message: str, data: dict) :
    response = {
        "success":success,
        "status_code":status,
        "message":message,
        "data":data
    }
    return response 

async def handle_exception(self, e: Exception) :
    pass