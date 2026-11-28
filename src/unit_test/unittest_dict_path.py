from __future__ import annotations


# Init startup path, change current path to test py file folder 
#-----------------------------------------------------------------
import os
from startup_init import startup_init_path
startup_init_path(os.path.dirname(os.path.abspath(__file__)))
#-----------------------------------------------------------------

from flex.ml_utils import get_dict_value, set_dict_value, DictPath

def test_extract_value():
    test_dict = {'foo1':{'foo2':{'foo3':{'foo4':'bar'}}}}
    assert get_dict_value(test_dict, 'foo1/foo2/foo3/foo4') == 'bar'

def test_extract_dict():
    test_dict = {'foo1':{'foo2':{'foo3':{'foo4':'bar'}}}}
    assert get_dict_value(test_dict, 'foo1/foo2/foo3') == {'foo4':'bar'}

def test_inject_value():
    test_dict = {'foo1':{'foo2':{'foo3':{}}}}
    set_dict_value(test_dict, 'foo1/foo2/foo3', {'foo4':'bar'})
    assert get_dict_value(test_dict, 'foo1/foo2/foo3/foo4') == 'bar'

def test_extract_array():
    test_dict = {'foo1':{'foo2':[{'foo3':'bar'},{'foo4':[[{'foo5': 'bar'}]]}]}}
    assert get_dict_value(test_dict, 'foo1/foo2/1/foo4/0/0/foo5') == 'bar'
    assert get_dict_value(test_dict, 'foo1/foo2/1/foo4/0/0/foo5/') == 'bar'

test_dict = {'foo1':{'foo2':{'foo3':{'foo4':'bar'}}}}
data = DictPath(test_dict)

def test_get_with_path():
    assert data.get('foo1/foo2/foo3/foo4') == 'bar'

def test_get_with_path_start_with_slash():
    assert data.get('/foo1/foo2/foo3/foo4') == 'bar'

def test_get_with_get():
    assert data.get('foo1').get('foo2').get('foo3') == {'foo4':'bar'}

def test_get_with_get1():
    assert data['foo1']['foo2']['foo3'] == {'foo4':'bar'}

def test_set_with_path():
    data.set('foo1/foo2/foo3/foo5', 'bar1')
    assert data.get('foo1/foo2/foo3/foo5') == 'bar1'

def test_set_with_path_start_with_slash():
    data.set('/foo1/foo2/foo3/foo5', 'bar1')
    assert data.get('foo1/foo2/foo3/foo5') == 'bar1'

def test_set_with_get():
    data.get('foo1').get('foo2').get('foo3')['foo5'] = 'bar1'
    assert data.get('foo1/foo2/foo3/foo5') == 'bar1'

def test_set_with_get1():
    data['foo1']['foo2']['foo3']['foo5'] = 'bar1'
    assert data.get('foo1/foo2/foo3/foo5') == 'bar1'

def test_equalness_with_normal_dict():
    normal_dict = {'foo1':{'foo2':{'foo3':{'foo4':'bar'}}}}
    dic_path = DictPath(normal_dict)
    assert normal_dict == dic_path
    assert dic_path.dict is normal_dict

def test_extract_array_with_dict():
    test_dict = {'foo1':{'foo2':[{'foo3':'bar'},{'foo4':[[{'foo5': 'bar'}]]}]}}
    dic_path = DictPath(test_dict)
    assert dic_path.get('foo1/foo2/1/foo4/0/0/foo5/') == 'bar'

def test_extract_array_with_dict_none():
    test_dict = {'foo1':{'foo2':[{'foo3':'bar'},{'foo4':[[{'foo5': 'bar'}]]}]}}
    dic_path = DictPath(test_dict)
    assert dic_path.get('foo1/foo2/1/foo4/0/0/foo6/foo7/') == None

if __name__ == "__main__":
    test_extract_value()
    test_extract_dict()
    test_inject_value()
    test_extract_array

    test_get_with_path()
    test_get_with_path_start_with_slash()
    test_get_with_get()
    test_get_with_get1()
    test_set_with_path()
    test_set_with_path_start_with_slash()
    test_set_with_get()
    test_set_with_get1()
    test_equalness_with_normal_dict()
    test_extract_array_with_dict()
    test_extract_array_with_dict_none()