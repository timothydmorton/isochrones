from isochrones.query import Query, TwoMASS, WISE, Tycho2

def test_queries(ra=45.03433035439128, dec=0.23539164875137225,
                    pmra=43.75231341609215, pmdec=-7.6419899883511482,
                    epoch=2015.):
    """Testing with first entry from Gaia DR1 TGAS table
    """
    q = Query(ra, dec, pmra=pmra, pmdec=pmdec, epoch=epoch)

    tm = TwoMASS(q)
    w = WISE(q)
    tyc = Tycho2(q)

    assert tm.get_id() == '03000819+0014074'
    assert w.get_id() == 'J030008.22+001407.4'
    assert tyc.get_id() == '55-256-1'

