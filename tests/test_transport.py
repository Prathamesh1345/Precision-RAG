from app.store import transport_url


def test_local_docker_connection_uses_ipv4_without_changing_remote_urls():
    assert transport_url('http://localhost:6333/path') == 'http://127.0.0.1:6333/path'
    assert transport_url('https://example.com:6333') == 'https://example.com:6333'
    assert transport_url('http://[::1]:6333') == 'http://[::1]:6333'
