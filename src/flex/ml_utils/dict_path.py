from __future__ import annotations
from collections import UserDict
from typing import Union, Any

import copy
import json


class DictPath(UserDict):
	"""
	DicPath is a normal dict that you can extract and inject the complex nested dict only using Path.
	"""

	def __init__(self, dictionary: dict = {}, is_deep_copy: bool = False):
		"""
		Accept python dict or DictPath.
		"""
		super().__init__(dictionary)
		if isinstance(dictionary, DictPath):
			self.data = dictionary.data
		elif isinstance(dictionary, dict):
			self.data = dictionary
		else:
			raise Exception("dictionary must be a dict")
		if is_deep_copy:
			self.data = copy.deepcopy(self.data)

	@property
	def dict(self) -> dict:
		"""
		Returns a reference to the dict.
		"""
		return self.data

	@property
	def deepcopy(self) -> DictPath:
		"""
		Create a complete copy of a DictPath
		"""
		return DictPath(self.data, is_deep_copy=True)

	def __repr__(self) -> str:
		"""
		Returns a pretty indented json string representation.
		"""
		dump = json.dumps(self.data, indent=2, sort_keys=True)
		return f"DictPath({dump})"

	def clean_path(self, path: str) -> list[str]:
		path = path[1:] if path.startswith('/') else path
		return path.split('/')

	def get(self, key: str, default=None):
		"""
		Get the value of the dictionary at the given path
		dict.get("foo/bar/foo1/bar1") is like calling dict["foo"]["bar"]["foo1"]["bar1"].
		Invalid paths return None without error.
		"""
		if key == "":
			return self
		
		current = self.data
		paths = self.clean_path(key)

		for attr in paths:
			if current is None:
				return default

			if isinstance(current, dict):
				current = current.get(attr)
			elif isinstance(current, list):
				current = current[int(attr)]

		if isinstance(current, dict):
			return DictPath(current)
		return current

	def set(self, path: str, value: Any=None):
		"""
		Set the value of the dictionary at the given path
		dict.set("foo/bar/foo1/bar1", 'bar') is like calling dict["foo"]["bar"]["foo1"]["bar1"] = "bar".
		If a path does not exist, it will be created.
		Empty path will do nothing.
		"""
		paths = self.clean_path(path)
		current = self.data
		last_path_attr = paths.pop()
		for attr in paths:
			if not isinstance(current, dict):
				raise Exception("Can't set the key of a non-dict")
			current.setdefault(attr, {})
			current = current[attr]
		if isinstance(value, DictPath):
			current[last_path_attr] = value.data
		else:
			current[last_path_attr] = value
		return

	# FIX wrong behavior of standard library
	# <a>.pop(<b>, None) returns key error, if <b> not in <a>.
	# This fixes it.
	def pop(self, key, *args):
		return self.data.pop(key, *args)

	def __getitem__(self, path) -> Any:
		""" Subscript for <DictPath>.get() """
		# If DictPath["key1"], then path="key1"
		# DictPath["key1", "key2"], then path=tuple("key1", "key2")

		path = "/".join(list(path)) if isinstance(path, tuple) else path
		return self.get(path)

	def __setitem__(self, path, value):
		""" Subscript for <DictPath>.get() and <DictPath>.apply_at_path() """
		path = "/".join(list(path)) if isinstance(path, tuple) else path
		self.set(path, value=value)


def get_dict_value(dictionary, path):
	path = path[1:] if path.startswith('/') else path
	paths = path.split('/')
	active_dict = dictionary

	for p in paths:
		if active_dict is None:
			return None
		if isinstance(active_dict, dict):
			active_dict = active_dict.get(p)
		elif isinstance(active_dict, list):
			active_dict = active_dict[int(p)]
	return active_dict


def set_dict_value(dictionary, path, value):
	path = path[1:] if path.startswith('/') else path
	paths = path.split('/')
	path_len = len(paths)

	_active_dict = dictionary
	for i, p in enumerate(paths):
		if i == path_len - 1:
			_active_dict[p] = value
			continue

		if _active_dict.get(p) is None:
			_active_dict[p] = {}
		_active_dict = _active_dict.get(p)
	return dictionary
