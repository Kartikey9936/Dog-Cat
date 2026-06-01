import traceback
try:
    from backend.model import model
    print('model input shape', model.input_shape)
    print('model output shape', model.output_shape)
    print('layers:')
    for i, layer in enumerate(model.layers):
        print(i, layer.name, layer.input_shape, layer.output_shape)
except Exception as e:
    traceback.print_exc()
    print('EXCEPTION TYPE:', type(e).__name__)
