# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: grapevine.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='grapevine.proto',
  package='grapevine',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n\x0fgrapevine.proto\x12\tgrapevine\"6\n\x07Message\x12\x0b\n\x03raw\x18\x01 \x01(\t\x12\x0c\n\x04text\x18\x02 \x01(\t\x12\x10\n\x08language\x18\x03 \x01(\t\"\x87\x01\n\x0e\x43lassification\x12\x0e\n\x06\x64omain\x18\x01 \x01(\t\x12\x12\n\nprediction\x18\x02 \x01(\t\x12\x12\n\nconfidence\x18\x03 \x01(\x01\x12\r\n\x05model\x18\x04 \x01(\t\x12\x0f\n\x07version\x18\x05 \x01(\t\x12\x1d\n\x04meta\x18\x06 \x01(\x0b\x32\x0f.grapevine.Meta\".\n\x04Meta\x12&\n\tsentences\x18\x01 \x03(\x0b\x32\x13.grapevine.Sentence\"F\n\x08Sentence\x12\x16\n\x0esentence_score\x18\x01 \x01(\x01\x12\x13\n\x0bword_scores\x18\x02 \x03(\x01\x12\r\n\x05words\x18\x03 \x03(\t\"]\n\nExtraction\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x0e\n\x06values\x18\x02 \x03(\t\x12\x12\n\nconfidence\x18\x03 \x01(\x01\x12\r\n\x05model\x18\x04 \x01(\t\x12\x0f\n\x07version\x18\x05 \x01(\t2I\n\nClassifier\x12;\n\x08\x43lassify\x12\x12.grapevine.Message\x1a\x19.grapevine.Classification\"\x00\x32\x43\n\tExtractor\x12\x36\n\x07\x45xtract\x12\x12.grapevine.Message\x1a\x15.grapevine.Extraction\"\x00\x62\x06proto3')
)




_MESSAGE = _descriptor.Descriptor(
  name='Message',
  full_name='grapevine.Message',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='raw', full_name='grapevine.Message.raw', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='text', full_name='grapevine.Message.text', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='language', full_name='grapevine.Message.language', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=30,
  serialized_end=84,
)


_CLASSIFICATION = _descriptor.Descriptor(
  name='Classification',
  full_name='grapevine.Classification',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='domain', full_name='grapevine.Classification.domain', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='prediction', full_name='grapevine.Classification.prediction', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='confidence', full_name='grapevine.Classification.confidence', index=2,
      number=3, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='model', full_name='grapevine.Classification.model', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='version', full_name='grapevine.Classification.version', index=4,
      number=5, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='meta', full_name='grapevine.Classification.meta', index=5,
      number=6, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=87,
  serialized_end=222,
)


_META = _descriptor.Descriptor(
  name='Meta',
  full_name='grapevine.Meta',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='sentences', full_name='grapevine.Meta.sentences', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=224,
  serialized_end=270,
)


_SENTENCE = _descriptor.Descriptor(
  name='Sentence',
  full_name='grapevine.Sentence',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='sentence_score', full_name='grapevine.Sentence.sentence_score', index=0,
      number=1, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='word_scores', full_name='grapevine.Sentence.word_scores', index=1,
      number=2, type=1, cpp_type=5, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='words', full_name='grapevine.Sentence.words', index=2,
      number=3, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=272,
  serialized_end=342,
)


_EXTRACTION = _descriptor.Descriptor(
  name='Extraction',
  full_name='grapevine.Extraction',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='grapevine.Extraction.key', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='values', full_name='grapevine.Extraction.values', index=1,
      number=2, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='confidence', full_name='grapevine.Extraction.confidence', index=2,
      number=3, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='model', full_name='grapevine.Extraction.model', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='version', full_name='grapevine.Extraction.version', index=4,
      number=5, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=344,
  serialized_end=437,
)

_CLASSIFICATION.fields_by_name['meta'].message_type = _META
_META.fields_by_name['sentences'].message_type = _SENTENCE
DESCRIPTOR.message_types_by_name['Message'] = _MESSAGE
DESCRIPTOR.message_types_by_name['Classification'] = _CLASSIFICATION
DESCRIPTOR.message_types_by_name['Meta'] = _META
DESCRIPTOR.message_types_by_name['Sentence'] = _SENTENCE
DESCRIPTOR.message_types_by_name['Extraction'] = _EXTRACTION
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Message = _reflection.GeneratedProtocolMessageType('Message', (_message.Message,), dict(
  DESCRIPTOR = _MESSAGE,
  __module__ = 'grapevine_pb2'
  # @@protoc_insertion_point(class_scope:grapevine.Message)
  ))
_sym_db.RegisterMessage(Message)

Classification = _reflection.GeneratedProtocolMessageType('Classification', (_message.Message,), dict(
  DESCRIPTOR = _CLASSIFICATION,
  __module__ = 'grapevine_pb2'
  # @@protoc_insertion_point(class_scope:grapevine.Classification)
  ))
_sym_db.RegisterMessage(Classification)

Meta = _reflection.GeneratedProtocolMessageType('Meta', (_message.Message,), dict(
  DESCRIPTOR = _META,
  __module__ = 'grapevine_pb2'
  # @@protoc_insertion_point(class_scope:grapevine.Meta)
  ))
_sym_db.RegisterMessage(Meta)

Sentence = _reflection.GeneratedProtocolMessageType('Sentence', (_message.Message,), dict(
  DESCRIPTOR = _SENTENCE,
  __module__ = 'grapevine_pb2'
  # @@protoc_insertion_point(class_scope:grapevine.Sentence)
  ))
_sym_db.RegisterMessage(Sentence)

Extraction = _reflection.GeneratedProtocolMessageType('Extraction', (_message.Message,), dict(
  DESCRIPTOR = _EXTRACTION,
  __module__ = 'grapevine_pb2'
  # @@protoc_insertion_point(class_scope:grapevine.Extraction)
  ))
_sym_db.RegisterMessage(Extraction)



_CLASSIFIER = _descriptor.ServiceDescriptor(
  name='Classifier',
  full_name='grapevine.Classifier',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  serialized_start=439,
  serialized_end=512,
  methods=[
  _descriptor.MethodDescriptor(
    name='Classify',
    full_name='grapevine.Classifier.Classify',
    index=0,
    containing_service=None,
    input_type=_MESSAGE,
    output_type=_CLASSIFICATION,
    serialized_options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_CLASSIFIER)

DESCRIPTOR.services_by_name['Classifier'] = _CLASSIFIER


_EXTRACTOR = _descriptor.ServiceDescriptor(
  name='Extractor',
  full_name='grapevine.Extractor',
  file=DESCRIPTOR,
  index=1,
  serialized_options=None,
  serialized_start=514,
  serialized_end=581,
  methods=[
  _descriptor.MethodDescriptor(
    name='Extract',
    full_name='grapevine.Extractor.Extract',
    index=0,
    containing_service=None,
    input_type=_MESSAGE,
    output_type=_EXTRACTION,
    serialized_options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_EXTRACTOR)

DESCRIPTOR.services_by_name['Extractor'] = _EXTRACTOR

# @@protoc_insertion_point(module_scope)