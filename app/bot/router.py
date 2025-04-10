





# for swagger
# @router.post("/")
# def upload_file(file: UploadFile, fileType: str = Form(...)):
#     filename = UPLOAD_FOLDER / file.filename # type: ignore
#     with open(filename, "wb") as f:
#         f.write(file.file.read())

#     # Логика обработки файла
#     # ...
#     create_vector_db(filename)  # type: ignore # Добавление в векторную базу

#     # Отправка уведомления в Telegram
#     send_telegram_message(f"Новый документ добавлен в векторную базу: {file.filename}")

#     return {"message": "Файл успешно обработан и добавлен в базу"}
