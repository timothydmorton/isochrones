from isochrones.query import Query, TwoMASS, WISE, Tycho2, Gaia

import sys
if sys.version_info < (3,):
    def b(x):
        return x
else:
    import codecs
    def b(x):
        return codecs.latin_1_encode(x)[0]

def test_queries(ra=45.03433035439128, dec=0.23539164875137225,
                    pmra=43.75231341609215, pmdec=-7.6419899883511482,
                    epoch=2015.):
    """Testing with first entry from Gaia DR1 TGAS table
    """
    q = Query(ra, dec, pmra=pmra, pmdec=pmdec, epoch=epoch)

    tm = TwoMASS(q)
    w = WISE(q)
    tyc = Tycho2(q)
    gaia = Gaia(q)

    assert tm.get_id() == b('03000819+0014074') #force byte literal b/c that's what gets returned
    assert w.get_id() == b('J030008.22+001407.4')
    assert tyc.get_id() == '55-256-1'
    assert gaia.get_id() == 7632157690368
