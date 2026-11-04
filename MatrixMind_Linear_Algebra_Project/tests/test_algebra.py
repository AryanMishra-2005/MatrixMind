from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.routes.algebra import router

app = FastAPI(); app.include_router(router, prefix="/api/v1")
client = TestClient(app)


def test_matrix_multiplication():
    r = client.post('/api/v1/algebra/calculate', json={
        'operation':'multiply','matrix_a':[[1,2],[3,4]],'matrix_b':[[5,6],[7,8]]})
    assert r.status_code == 200
    assert r.json()['result'] == [[19,22],[43,50]]


def test_rref_and_rank():
    r = client.post('/api/v1/algebra/calculate', json={
        'operation':'rref','matrix_a':[[1,2,3],[2,4,6]]})
    assert r.status_code == 200
    assert r.json()['result'] == [[1,2,3],[0,0,0]]


def test_dimension_validation():
    r = client.post('/api/v1/algebra/calculate', json={
        'operation':'multiply','matrix_a':[[1,2]],'matrix_b':[[1,2]]})
    assert r.status_code == 400
