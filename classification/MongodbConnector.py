import pymongo


client = pymongo.MongoClient("mongodb://localhost:27017/")

print(client.list_database_names())

ia_db = client["ia"]
ia_data = ia_db["data"]

for item in ia_data.find():
    print(item)


def insert_in_db(tag, text):
    temp_result = ia_data.find_one({
        "tag": tag,
        "patterns": text
    })

    if temp_result is None:
        result = ia_data.update_one(
            {
                "tag": tag
            },
            {
                "$push": {"patterns": text}
            }
        )

        print("results affected:", result.matched_count)

