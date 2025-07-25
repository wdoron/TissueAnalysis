
class JsonInfo:
    def __init__(self):
        self._json = {}

    def get_json(self):
        return self._json

    def add_to_json(self, data: any, category: str, sub_category: str | None = None, id: str | int | None = None,
                    should_print: bool = False):

        if sub_category is None:
            sub_category = "NO_SUB_CATEGORY"
        if id is None:
            id = "NO_ID"

        if category not in self._json or not isinstance(self._json.get(category), dict):
            self._json[category] = {}

        if sub_category not in self._json[category] or not isinstance(self._json[category].get(sub_category), dict):
            self._json[category][sub_category] = {}

        if self._json[category][sub_category].get(id) is None:
            self._json[category][sub_category][id] = []

        self._json[category][sub_category][id].append(data)

        if should_print:
            print(f"{category}.{sub_category}.{id} = {data}")

