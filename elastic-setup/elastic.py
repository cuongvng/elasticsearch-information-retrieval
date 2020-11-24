from elasticsearch import Elasticsearch
from elasticsearch.exceptions import ConnectionError, NotFoundError
from elasticsearch.helpers import bulk
import numpy as np
import json

class Elastic(object):
	def __init__(self):
		self.client = Elasticsearch('localhost:9200')

	def create_index(self, name, config_file):
		try:
			with open(config_file) as file:
				config = json.load(file)

			self.client.indices.create(index=name, body=config)
			print(f"[INFO] index {name} has been created!")
		except Exception as e:
			print(e)

	def delete_index(self, name):
		try:
			self.client.indices.delete(index=name)
		except NotFoundError:
			print(f"Index {name} not found!")

