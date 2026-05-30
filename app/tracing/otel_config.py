from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

def setup_tracing(app, service_name="the-app"):
    resource = Resource(attributes={
        "service.name": service_name
    })

    otlp_exporter = OTLPSpanExporter(
        endpoint="otel-collector:4317",
        insecure=True
    )

    trace.set_tracer_provider(TracerProvider(resource=resource))
    
    span_processor = BatchSpanProcessor(otlp_exporter)
    trace.get_tracer_provider().add_span_processor(span_processor)
    
    FastAPIInstrumentor.instrument_app(app)
    
    return trace.get_tracer(__name__)

tracer = trace.get_tracer("nlp-mlops-tracer")
